import pandas as pd

import logging
logger = logging.getLogger(f'cibi.{__file__}')

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
                       deduplication=True,
                       save_file=None,
                       flush_every=20):
        self.metrics = metrics
        self.metadata = metadata
        self.save_file = save_file
        self.deduplication = deduplication

        assert type(flush_every) == int and flush_every > 0
        self.flush_every = flush_every
        self.flush_ttl = flush_every

        columns = ['code', 'count'] + metrics + metadata
        types = [object, int] + [float for m in metrics] + [object for m in metadata]

        if self.save_file:
            try:
                cache = pd.read_pickle(save_file)
                assert list(cache.columns) == columns

                self.data_frame = cache.astype({
                    cname: ctype for cname, ctype in zip(columns, types)
                })
            except FileNotFoundError:
                self.data_frame = None
        else:
            self.data_frame = None
                
        if self.data_frame is None:
            self.data_frame = make_dataframe(columns=columns, 
                                             dtypes=types, 
                                             index_column='code' if deduplication else None)

    def commit(self, code, metrics={}, metadata={}, count=1):
        def append_row(row_count):
            new_row = {
                'code': code,
                'count': row_count,
                **metrics,
                **metadata
            }
            self.data_frame = self.data_frame.append(pd.Series(name=code,data=new_row))

        if self.deduplication:
            try:
                program_row = self.data_frame.loc[code]
                program_count = program_row['count']

                for metric in self.metrics:
                    try:
                        # We store mean metrics over all occurences of the program
                        old_metric = program_row[metric]
                        metric_update = metrics[metric]
                        if old_metric != old_metric:
                            program_row[metric] = metric_update
                        else:
                            program_row[metric] = ((old_metric * program_count + metric_update) 
                                                / (program_count + count))
                    except KeyError:
                        pass
                program_row['count'] = program_count + count

                for metadata_column in self.metadata:
                    try:
                        # We store metadata only for the first occurence of the program
                        if program_row[metadata_column] != program_row[metadata_column]:
                            program_row[metadata_column] = metadata[metadata_column]
                    except KeyError:
                        pass

                self.data_frame.loc[code] = program_row
            except KeyError:
                append_row(count)
        else:    
            for x in range(count):
                append_row(1)

        if self.flush_ttl == 0:
            self.flush()
            self.flush_ttl = self.flush_every
        else:
            self.flush_ttl -= 1

    def merge(self, other_codebase, force=False):
        if not force:
            assert self.metrics == other_codebase.metrics
            assert self.metadata == other_codebase.metadata

        for _, row in other_codebase.data_frame.iterrows():
            metrics = {metric: row[metric] for metric in self.metrics}
            metadata = {meta: row[meta] for meta in self.metadata}
            if self.deduplication:
                self.commit(row['code'], metrics=metrics, 
                                         metadata=metadata, 
                                         count=row['count'])
            else:
                for _ in range(row['count']):
                    self.commit(row['code'], metrics=metrics, 
                                             metadata=metadata, 
                                             count = 1)

    def query(self, expr):
        subcodebase = make_codebase_like(self)
        subcodebase.data_frame = self.data_frame.query(expr)
        return subcodebase

    def replace(self, other_codebase):
        assert self.metrics == other_codebase.metrics
        assert self.metadata == other_codebase.metadata
        assert other_codebase.deduplication

        for code, data in other_codebase.data_frame.iterrows():
            self.data_frame.replace(self.data_frame['code'] == code, data, inplace=True)

    def subset(self, codes):
        subcodebase = make_codebase_like(self)
        subcodebase.data_frame = self.data_frame.loc[codes]
        return subcodebase

    def top_k(self, metric, k=3):
        assert metric in self.metrics, f'{metric} column not present in this codebase'

        sampled_codebase = make_codebase_like(self)
        sampled_codebase.data_frame = self.data_frame.nlargest(k, metric)
        return sampled_codebase

    def __getitem__(self, column):
        return list(self.data_frame[column])

    def __setitem__(self, column, value):
        self.data_frame[column] = value

    def __len__(self):
        return len(self.data_frame.index)

    def clear(self):
        self.data_frame = self.data_frame.iloc[0:0]

    def sample(self, n=1, metric=None, keep_count=False):
        sample_from = self.data_frame
        weights = None

        if metric:
            weights = self.data_frame[metric] 
            sample_from = self.data_frame[weights != 0]

        sample_size = min(n, len(sample_from))
        
        sampled_data_frame = sample_from.sample(n=sample_size, weights=weights)

        sampled_codebase = make_codebase_like(self)
        sampled_codebase.data_frame = sampled_data_frame
        sampled_codebase.deduplication = True

        if not keep_count:
            sampled_codebase['count'] = 1

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

    def to_string(self):
        # https://stackoverflow.com/questions/55755695/python-3-6-logger-to-log-pandas-dataframe-how-to-indent-the-entire-dataframe/55770434
        return '\t'+ self.data_frame.to_string().replace('\n', '\n\t') 

    def flush(self):
        if self.save_file:
            self.data_frame.to_pickle(self.save_file)

def make_dev_codebase(save_file=None):
    return Codebase(metrics=['log_prob'],
                    metadata=['method', 'parent1', 'parent2'],
                    deduplication=False,
                    save_file=save_file)

def make_prod_codebase(deduplication, save_file=None):
    return Codebase(metrics=['total_reward', 'quality', 'log_prob'],
                    metadata=['result', 'author', 'method', 'parent1', 'parent2'],
                    deduplication=deduplication,
                    save_file=save_file)

def make_codebase_like(other_codebase):
    c = Codebase(metrics=other_codebase.metrics,
                 metadata=other_codebase.metadata,
                 deduplication=other_codebase.deduplication)
    return c