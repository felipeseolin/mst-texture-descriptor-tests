from TestBase import TestBase


def main():
    test_base = TestBase()
    test_base.set_decision_tree_classifier()
    test_base.set_lbp_descriptor()
    test_base.set_brodatz_dataset()
    test_base.test_model()


if __name__:
    main()
