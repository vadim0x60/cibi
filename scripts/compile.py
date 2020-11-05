import click

@click.command('Transform a development file to a BF++ program bu removing comments')
@click.argument('dev-file')
def compile(dev_file):
    assert dev_file[-8:] == '_dev.txt'

    with open(dev_file, 'r') as dev_f:
        with open(dev_file[:-8] + '.txt', 'w') as out_f:
            code = ''
            def new_program():
                nonlocal code
                if code:
                    out_f.write(code + '\n')
                code = ''

            new_program()

            for line in dev_f.readlines():
                if not line or line[0] == '#':
                    continue

                if line[:3] == '===':
                    new_program()
                    continue

                code += line.strip()
            
            new_program()

if __name__ == '__main__':
    compile()