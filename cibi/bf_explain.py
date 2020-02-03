from cibi import bf
import re

cell_actions = ''.join(bf.SHORTHAND_CELLS) + '><'

def prune(code):
    return re.sub(f'[{cell_actions}]+(?=[{bf.SHORTHAND_CELLS}])', '', code)

def explain(code):
    explained_code = None
    return explained_code

if __name__ == '__main__':
    print(prune('>>>>a'))
    print(prune('>>bd<<'))