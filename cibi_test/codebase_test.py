import tensorflow as tf
from cibi.codebase import make_prod_codebase

class CodebaseTest(tf.test.TestCase):
    def testDeduplication(self):
        codebase = make_prod_codebase(deduplication=True)

        codebase.commit('+>+', metrics={'total_reward': 0})
        codebase.commit('+>+', metrics={'total_reward': 10})
        
        self.assertEqual(len(codebase), 1)
        self.assertEqual(codebase['count'][0], 2)
        self.assertEqual(codebase['code'][0], '+>+')
        self.assertEqual(codebase['total_reward'][0], 5)

if __name__ == '__main__':
  tf.test.main()