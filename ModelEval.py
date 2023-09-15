import os.path

from fairseq.criterions import FairseqCriterion
from fairseq.models.transformer import TransformerModel

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.scoring import BaseScorer, register_scorer


def evaluate():
    model_name = "transformer"
    data_set = "C:\\Users\\MKhal\\OneDrive\\Desktop\\UNI\\Bachelorarbeit\\DEV\\generator_out\\1D-Bracket"

    model = TransformerModel.from_pretrained(
        model_name_or_path=os.path.join(data_set, "checkpoints"),
        checkpoint_file="checkpoint_best.pt",
        data_name_or_path=os.path.join(data_set, "data-bin")
    )


@dataclass
class AccuracyScorerConfig(FairseqDataclass):
    pass


def str_accuracy(label: str, pred: str) -> float:
    ls = label.split(" ")
    ps = pred.split(" ")
    hits = 0
    for l, p in zip(ls, ps[:len(ls)]):
        hits += int(l == p)
    return hits / len(ls)


@register_scorer("accuracy_score", dataclass=AccuracyScorerConfig)
class AccuracyScorer(BaseScorer):

    def score(self) -> float:
        return sum(map(lambda pair: str_accuracy(*pair), zip(self.ref, self.pred))) / len(self.ref)

    def result_string(self) -> str:
        return f"Accuracy : {self.score()}"
