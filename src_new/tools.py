from ast import Pass
import logging
from unittest import result
from langchain.agents import AgentType, initialize_agent
from langchain.chains import LLMMathChain
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
from langchain.utilities import SerpAPIWrapper
from langchain.docstore.document import Document
from .utils import Node, NoteBook
from typing import List, Union
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

#TODO: If necessary, add pydantic, refer to BingAPI


class TreeAPI:
    def __init__(self, tree: Node) -> None:
        self.k = []
        self.tree = tree
        self.pos = self.tree
        import dotenv
        dotenv.load_dotenv('./src_new/azure.env')
        self.embeddings_model = AzureOpenAIEmbeddings(deployment="text-embedding-ada-002")
    
    def get_parent(self):
        cur = self.tree
        if len(self.k)>1:
            for k in self.k[:-1]:
                cur=cur['children'][k]
        return cur
    
    def reset(self):
        self.k = []
        self.pos = self.tree

import json
tree = json.load(open('/workspace/yule/PDFTriage/haigong/test.json'))
cls = TreeAPI(tree)

notebook = NoteBook()

@tool('get_children')
def get_children(): 
    '''  
    Retrieves children nodes of the current position of the tree.
    Args:
        None
    Returns:
        result: The content of  the children nodes
    '''
    ctx = [node['text'] for node in cls.pos['children']]
    if len(ctx)==0:
        return 'There is no children nodes.'
    if len(ctx)>20 and cls.pos['path']!='//Document':
        return 'There are over 20 children of current node, you should use the `search_given_children` tool with k=all to get the top4 relevant content.'
    else:
        result = 'The results are formated by {index}:{content}. '
        for index, string in enumerate(ctx):
            result += f"{index}: {string} "
        return result

@tool('take_note')
def take_note(text):
    '''
    If you found some useful information and need more, you can write current information on the notebook by using this tool.
    Remember the action name is **write_note**
    Args: 
        text: The information you want to remember
    Returns:
        
    '''
    notebook.write(text)
    return f'You have written down the information, you have {notebook.length()} notes now.'

@tool('look_note')
def look_note():
    '''
    Before you draw a finial conclusion, you **should** lookup the notebook first by using `look_note`
    Args:
    Returns:
        notes: All the information you have written down
    '''
    return notebook.read()

@tool('search_given_children')
def search_given_children(k:Union[List[int],str], text):
    '''
    If you are not sure which child will contain the given text, you can use this tool to search which child will have the related information.\
    If the given node have children nodes, the tool will search the children of the given child-node to determine the relation. Or it only considers the given nodes.\
    If there are many children nodes, use `all` mode to search **before** you decide on yourself. You should vist **at least** the top-2 choices to draw a conclusion
    Args: 
        k: The indices of children you want to query, it can be List[int] or str('all').
    Returns:
        result:dict : The top four relevant nodes you get. This function returns the {(index1, index2): score} dict.
        If index2 is not none, it means the index2-th child in the index1-th child of  current node contains the target data
        If index2 is none, it means the index1-th child of current node contains the target data
    
    Examples:
        1. You have one candidate of the child(k1) and want to know if this child  contains the given text
            use 
            ```
            search_given_children([k1], text)
            ```
        2. You have several candidates of the children (k1,...,kn) and want to know which child contains the given text
            use 
            ```
            search_given_children([k1,...,kn], text)
            ```
        3. Since there are too many children of a single node, you cannot read them at once. You call this function to do embedding reseach.
            use 
            ```
            search_given_children('all', text)
            ```
            You should vist **at least** the top-2 choices to draw a conclusion
    '''
    if k == 'all':
        k = [ki for ki in range(len(cls.pos['children']))]
    ctx=[]
    for k_i in k:
        if cls.pos['children'][k_i]['children']:
            ctx.extend([Document(page_content=node['text'], metadata={'source':f'({k_i}, {index})'}) for index, node in enumerate(cls.pos['children'][k_i]['children'])])
        else:
            ctx.append(Document(page_content=cls.pos['children'][k_i]['text'], metadata={'source':f'({k_i},)'}))
    db = Chroma.from_documents(documents=ctx, embedding=cls.embeddings_model)
    docs = db.similarity_search_with_relevance_scores(text,kwargs = {'score_threshold':0.5})
    result = {}
    for doc, score in docs:
        result[doc.metadata['source']] = score
    return result

@tool('move_to_kth_child')
def move_to_kth_child(k):
    '''
    Move to the k-th child of the current node.
    Args: 
        k:int :The k-th child, k should be smaller than the number of current children
    Returns:
        result: The content of  the k-th children nodes
    '''
    try:
        cls.pos = cls.pos['children'][k]
        cls.k.append(k)
        ctx = cls.pos['text']
        return f'You have moved to the {k}-th child node, the content is : {ctx}'
    except:
        return f"There are problem with the given index, the feasible index is from 0 to {len(cls.pos['children'])-1}"

@tool('get_current_position')
def get_current_position():
    '''
    Get the information of current node.
    Args: 
        None
    Returns:
        result: The content of  the current information
    '''
    # TODO: current path is
    if len(cls.pos['children'])==0:
        return f"Text: {cls.pos['text']} You can use `move_to_parent` to move to the parent node only because there is no children node no."
    else:
        return f"Text: {cls.pos['text']} You can now use `get_children` to acess the content of the children nodes \
        or use `move_to_kth_child` to move the child and access its children nodes or use `move_to_parent` to move to the parent node."


@tool('move_to_parent')
def move_to_parent():
    '''
    If you **have** looked through the content nodes and think the related information is **not** contained in the children nodes or itcls, \
    then you can use this tool to go back to the parent node and choose another node to access.
    Args: 
        None
    Returns:
        result: The content of  the parent node.
    '''
    if not cls.k:
        return 'You should only call this tool after you have moved inside the tree, you are at the top node now'
    else:
        cls.pos = cls.get_parent()
        last_visited = cls.k.pop(-1)
        return f"You have moved to the parent node, the index of the visited node is {last_visited}"
    
      


def return_tools():
    return [get_children, search_given_children, get_current_position,move_to_kth_child,move_to_parent,take_note,look_note]

def reset_tool():
    cls.reset()
    notebook.earse()