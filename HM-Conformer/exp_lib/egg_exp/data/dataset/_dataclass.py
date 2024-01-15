from dataclasses import dataclass

@dataclass
class SV_TrainItem:
    path: str
    speaker: str
    label: int

@dataclass
class SV_EnrollmentItem:
    key: str
    path: str

@dataclass
class SV_Trial:
    key1: str
    key2: str
    label: int

@dataclass
class DF_Item:
    path: str
    label: int
    attack_type: str
    is_fake: bool