import os
import argparse
#Apache Beam lee ficheros a gran escala
import apache_beam as beam
import tensorflow_transform.beam as tft_beam
import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.public import tfxio
from tfx_bsl.coders.example_coder import RecordBatchToExamples
from tensorflow_transform.tf_metadata import dataset_metadata, schema_utils
from apache_beam import PCollection, Pipeline
from typing import List, Dict
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions

def get_train_and_test(p: Pipeline, data_location: str) -> (PCollection[Dict], PCollection[Dict]):
    # p | "Soy una etiqueta" >> beam.Map() | "Otra etiqueta" >> beam.Al
    # p.apply(transform=beam.Map(), label="Soy una etiqueta") #esto es similar a lo de arriba

    # Tenemos que leer los positivos por un lado y los negativos por otro
    train_pos_location = os.path.join(data_location, 'train/pos/')
    train_neg_location = os.path.join(data_location, 'train/neg/')
    test_pos_location = os.path.join(data_location, 'test/pos/')
    test_neg_location = os.path.join(data_location, 'test/neg/')

    # Lectura del Apache Beam normal - Esto son líneas de texto
    train_pos: PCollection[str] = p | "Train pos" >> beam.io.ReadFromText(file_apttern=train_pos_location)
    train_neg: PCollection[str] = p | "Train neg" >> beam.io.ReadFromText(file_apttern=train_neg_location)

    test_pos: PCollection[str] = p | "Test pos" >> beam.io.ReadFromText(file_apttern=test_pos_location)
    test_neg: PCollection[str] = p | "Test neg" >> beam.io.ReadFromText(file_apttern=test_neg_location)

    train_pos_dicts: PCollection[Dict] = train_pos | "Add label train pos" >> beam.Map(
        lambda t: {'text': t, 'target': 1})
    train_neg_dicts: PCollection[Dict] = train_neg | "Add label train neg" >> beam.Map(
        lambda t: {'text': t, 'target': 0})

    test_pos_dicts: PCollection[Dict] = test_pos | "Add label test pos" >> beam.Map(
        lambda t: {'text': t, 'target': 1})
    test_neg_dicts: PCollection[Dict] = test_neg | "Add label test neg" >> beam.Map(
        lambda t: {'text': t, 'target': 0})

    # Juntar los ficheros en una sola colección para test y train
    train_dicts: PCollection[Dict] = (train_pos_dicts, train_neg_dicts) | "Train set" >> beam.Flatten()
    test_dicts: PCollection[Dict] = (test_pos_dicts, test_neg_dicts) | "Test set" >> beam.Flatten()

    return train_dicts, test_dicts

#Importante es importante convertir los ficheros a TFRecords que facilita el trabajo con TensorFlow

#Ejemplo de Preprocessing. Vamos a usar la simple que está debajo de esta
def preprocessing_fn_tfidf(inputs):

    texts = inputs['text']
    targets = inputs['target']

    #Vamos a separar las palabras
    words = tf.strings.split(texts, sep=' ').to_sparse()
    #Esto hace que el OneHotEncoding que solo almacena lo que importa para no tener muchos 0s y ser más optimo
    ngrams = tft.ngrams(words, ngram_range=(1, 2), separator=' ') #Ngramas de 1 y de 2
    vocabulary = tft.compute_and_apply_vocabulary(ngrams, top_k=20000)
    indices, weights = tft.tfidf(vocabulary, 20000)

    return {'indices': indices,
            'weights': weights,
            'targets': targets}

def preprocessing_fn(inputs):

    #texts = inputs['text']
    #targets = inputs['target']

    #Parece que no he hecho nada pero hemos hecho una transformación en TF_records
    outputs = inputs.copy()

    return outputs

def apply_tensorflow_transform(train_set: PCollection[Dict], test_set: PCollection[Dict], metadata):

    transf_train_ds, transform_fn = (train_set, metadata) | "TFT train" >> tft_beam.AnalyzeAndTransformDataset(
        preprocessing_fn=preprocessing_fn,
        output_record_batches=True)
    transf_train_pcoll, transf_train_metadata = transf_train_ds  # no nos hace falta el train_metadata

    # Lo que se hace en train, se aplica en test. La chicha está en el preprocessing
    test_set_ds = (test_set, metadata)
    transf_test_ds = (test_set_ds, transform_fn) | "TFT test" >> tft_beam.AnalyzeAndTransformDataset(
        output_record_batches=True)
    transf_test_pcoll, transf_test_metadata = transf_test_ds

    return transf_train_pcoll, transf_test_pcoll, transform_fn

def run_pipeline(argv: List[str], data_location: str, output_location: str):

    feature_spec = {
        'text': tf.io.FixedLenFeature([], tf.strings),
        'target': tf.io.FixedLenFeature([], tf.int64)
    }
    metadata = dataset_metadata.DatasetMetadata(
        schema_utils.schema_as_feature_spec(feature_spec)
    )

    options = PipelineOptions(argv)
    gcp_options = options.view_as(GoogleCloudOptions)
    temp_dir = gcp_options.temp_location #Recuperamos la dirección como parámetro
    train_output_location = os.path.join(output_location, 'train/')
    test_output_location = os.path.join(output_location, 'test/')

    #Lee los parámetros que se pasan como variables
    with beam.Pipeline(options=options) as p, tft_beam.Context(temp_dir=temp_dir): #El temp_dir tiene que ser en el cloud
        train_set, test_set = get_train_and_test(p, data_location)
        train_set_transf, test_set_transf, transform_fn = apply_tensorflow_transform(train_set, test_set, metadata)

        train_set_tf_example: PCollection[tf.train.Example] = train_set_transf | 'Train to example' >> beam.FlatMap(
            lambda r, _: RecordBatchToExamples(r))
        train_set_tf_example | 'Write train' >> beam.io.WriteToTFRecord(
            file_path_prefix=train_output_location,
            file_name_suffix='.tfrecord'
        )

        test_set_tf_example: PCollection[tf.train.Example] = test_set_transf | 'Test to example' >> beam.FlatMap(
            lambda r, _: RecordBatchToExamples(r))
        test_set_tf_example | 'Write train' >> beam.io.WriteToTFRecord(
            file_path_prefix=test_output_location,
            file_name_suffix='.tfrecord'
        )

        transform_fn_location = os.path.join(output_location, 'transform_fn/')
        transform_fn | "Write transform fn" >> tft_beam.WriteTransformFn(transform_fn_location)

    if __name__ == 'main':
        #Parsear los argumentos
        parser = argparse.ArgumentParser()

        parser.add_argument("--data-location", required=True)
        parser.add_argument("--output_location", required=True)

        known_args, other_args = parser.parse_known_args()

        data_location = known_args.data_location
        output_location = known_args.output_location

        run_pipeline(other_args, data_location, output_location)








