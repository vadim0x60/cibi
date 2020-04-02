import pandas as pd

def make_dataframe(columns, dtypes, index_column=None):
    # Stackoverflow-driven development (SDD) powered by 
    # https://stackoverflow.com/questions/36462257/create-empty-dataframe-in-pandas-specifying-column-types

    assert len(columns)==len(dtypes)
    df = pd.DataFrame()
    for c,d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    if index_column:
        df.set_index(index_column)
    return df

class Codebase():
    """
    A data structure for append-only storage of programs and their quality metrics

    It's just a pandas dataframe at the moment, but makes for an easy drop-in replacement
    with a more efficient implementation if need be
    """

    def __init__(self, metrics=[], 
                       metadata=[],  
                       save_file=None,
                       flush_every=20):
        self.metrics = metrics
        self.metadata = metadata
        self.save_file = save_file

        assert type(flush_every) == int and flush_every > 0
        self.flush_every = flush_every
        self.flush_ttl = flush_every

        columns = ['code', 'count'] + metrics + metadata
        types = [object, int] + [float for m in metrics] + [object for m in metadata]

        try:
            assert self.save_file
            self.data_frame = pd.read_pickle(save_file)
            assert self.data_frame.columns == columns
            assert self.data_frame.dtypes == types
        except (FileNotFoundError, AssertionError):
            self.data_frame = make_dataframe(columns=columns, 
                                             dtypes=types, 
                                             index_column='code')

    def commit(self, code, metrics={}, metadata={}):
        try:
            program_row = self.data_frame.loc[code]
            program_count = program_row['count']

            for metric in self.metrics:
                try:
                    # We store mean metrics over all occurences of the program
                    program_row[metric] = ((program_row[metric] * program_count + metrics[metric]) 
                                          / (program_count + 1))
                except KeyError:
                    pass
            program_row['count'] = program_count + 1

            for metadata_column in self.metadata:
                try:
                    # We store metadata only for the last occurence of the program
                    program_row[metadata_column] = metadata[metadata_column]
                except KeyError:
                    pass
        except KeyError:
            new_row = {
                'code': code,
                'count': 1,
                **metrics,
                **metadata
            }
            self.data_frame = self.data_frame.append(pd.Series(name=code,data=new_row))

        if self.flush_ttl == 0:
            self.flush()
            self.flush_ttl = self.flush_every
        else:
            self.flush_ttl -= 1

    def merge(self, other_codebase):
        metric_err = f'Failed to merge codebase with metris {other_codebase.metrics} into one with {self.metrics}'
        metadata_err = f'Failed to merge codebase with metadata {other_codebase.metadata} into one with {self.metadata}'
        
        assert self.metrics == other_codebase.metrics, metric_err
        assert self.metadata == other_codebase.metadata, metadata_err
        
        self.data_frame = self.data_frame.append(other_codebase.data_frame)

    def top_k(self, metric, k=3):
        assert metric in self.metrics
        return self.data_frame.nlargest(k, metric)

    def __getitem__(self, column):
        return self.data_frame[column]

    def __len__(self):
        return len(self.data_frame.index)

    def clear(self):
        self.data_frame = self.data_frame.iloc[0:0]

    def sample(self, n=1, metric=None):
        if metric:
            assert metric in self.metrics
        
        sampled_codebase = Codebase(metrics=self.metrics,
                                    metadata=self.metadata)
        sampled_codebase.data_frame = self.data_frame.sample(n=n, weights=metric)
        return sampled_codebase

    def peek(self):
        program = self.data_frame.iloc[0]

        code = program.name
        metrics = {metric: program[metric] for metric in self.metrics}
        metadata = {m: program[m] for m in self.metadata}
        return code, metrics, metadata

    def pop(self):
        r = self.peek()
        self.data_frame = self.data_frame.iloc[1:]
        return r

    def flush(self):
        if self.save_file:
            self.data_frame.to_pickle(self.save_file)

def make_dev_codebase():
    return Codebase(metrics=['value_estimate'],
                    metadata=['log_probs'])

def make_prod_codebase():
    return Codebase(metrics=['value_estimate', 'test_quality', 'replay_weight'],
                    metadata=['log_probs', 'result'])

if __name__ == '__main__':
    codebase = Codebase()
    codebase.commit('1>2!')
    print(codebase.sample(1).data_frame)
    codebase.clear()