import sys
sys.path.append('/mnt/5022A63622A620C8/TCC/tests')
sys.path.append('/mnt/5022A63622A620C8/TCC/tests/util')
sys.path.append('/mnt/5022A63622A620C8/TCC/tests/**')

from testBase import TestBase

def main():
    test_base = TestBase()
    test_base.set_decision_tree_classifier()
    test_base.set_gabor_descriptor()
    test_base.set_kylberg_dataset()
    test_base.test_model()


if __name__:
    main()
