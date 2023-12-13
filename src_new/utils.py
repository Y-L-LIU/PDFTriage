from typing import List
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Node:
    path: str
    text: str
    depth: int
    children=[]

class NoteBook:
    def __init__(self) -> None:
        self.notes=[]

    def length(self):
        return len(self.notes)

    def write(self,text):
        self.notes.append(text)

    def read(self):
        return '***'.join(self.notes)
    
    def earse(self):
        self.notes = []