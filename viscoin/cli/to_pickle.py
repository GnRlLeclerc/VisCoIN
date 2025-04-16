import click

from viscoin.cli.utils import checkpoints
from viscoin.models.classifiers import Classifier
from viscoin.models.concept_extractors import ConceptExtractor
from viscoin.models.explainers import Explainer
from viscoin.models.gan import GeneratorAdapted
from viscoin.models.utils import load_viscoin, save_viscoin_pickle


@click.command()
@checkpoints
@click.option("--output", help="The path to generate the pickle to", type=str)
def to_pickle(checkpoints: str, output: str):
    """Convert safetensors to a pickled viscoin model using default parameters"""

    classifier = Classifier()
    concept_extractor = ConceptExtractor()
    explainer = Explainer()
    gan = GeneratorAdapted()

    load_viscoin(classifier, concept_extractor, explainer, gan, checkpoints)
    save_viscoin_pickle(classifier, concept_extractor, explainer, gan, output)
