import click

@click.command('Transform a development file to a BF++ program bu removing comments')
@click.argument('dev-file')
def compile(dev_file):
    assert dev_file[-8:] == '_dev.txt'

    code = ''

    with open(dev_file, 'r') as dev_f:
        for line in dev_f.readlines():  
            if line and line[0] != '#':
                code += line.strip()
                    
    with open(dev_file[:-8] + '.txt', 'w') as out_f:
        out_f.write(code)

if __name__ == '__main__':
    compile()