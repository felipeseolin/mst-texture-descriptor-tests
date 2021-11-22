from TestBase import TestBase


def main():
    test_base = TestBase()
    test_base.set_linear_svc_classifier()
    test_base.set_gabor_descriptor()
    test_base.set_usptex_dataset()
    test_base.test_model()


if __name__:
    main()
