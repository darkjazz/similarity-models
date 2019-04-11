import json, couchdb

features = ['chords', 'combined', 'rhythm', 'timbre']
similarities = ['heat-prob-0', 'heat-prob-1', 'max', 'rank']

class Eval:
    def __init__(self):
        srv = couchdb.Server()
        self.dbs = { }
        self.dbs['eval_db'] = srv['eval_db']
        for _sim in similarities:
            for _f in features:
                _name = 'ab_db_%s_%s' % (_f, _sim)
                self.dbs[_name] = srv[_name]
