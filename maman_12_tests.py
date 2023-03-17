import unittest

VERSION = 0.1

environment = {
    "function_dist_valid":  is_distribution_valid,  #  input: parameter dist of my_sampler. output: True iff distribution is valid else False
    "test_dist_valid": False,
    "test_my_sampler": True
}

@unittest.skipIf(reason="chose to skip", condition=(environment["test_dist_valid"] == False))
class TestDistValid(unittest.TestCase):
    def test_dist_valid(self):
        bad_distributions = [
            [0.5, 0.4999999], # sum < 1
            [0.5, 0.5000001], #  sum > 1
            [-0.4, 0.3, 0.5, 0.6], # negative probability
            [0.2, 0, 0.5, 0.3], # zero probability
        ]
        good_distributions = [
            [0.5, 0.5],
            [0.1, 0.2, 0.7],
            [0.4, 0.4, 0.2],
            [10 ** -1 for _ in range(10)], # floating point error might occur in sum, fsum solves this
            [10 ** -5 for _ in range(10 ** 5)]
        ]
        dist_validator = environment["function_dist_valid"]
        for i, dist in enumerate(bad_distributions):
            self.assertEqual(dist_validator(dist), False, msg=f"expected True for bad distribution #{i} {dist}")
        for i, dist in enumerate(good_distributions):
            self.assertEqual(dist_validator(dist), True, msg=f"returned False for good distribution #{i} {dist}")

@unittest.skipIf(reason="chose to skip", condition=(environment["test_my_sampler"] == False))
class TestMySampler(unittest.TestCase):
    def test_my_sampler_binomial(self):
        epsilon = 0.002
        numel = 10 ** 7
        binomial_distribution = my_sampler(numel , [0.5, 0.5])
        binomial_sum = torch.sum(binomial_distribution)
        self.assertTrue(
            numel * 0.5 * (1 - epsilon) <= binomial_sum <=  numel * 0.5 * (1 + epsilon),
            msg=f"The expected value is {numel * 0.5}. Your result {binomial_sum} came out too far"
        )

    def test_my_sampler_uniform(self):
        epsilon = 0.05
        numel = 10 ** 4
        uniform_distribution = my_sampler((100, 100), [1 / numel for _ in range(numel)])
        uniform_sum = torch.sum(uniform_distribution)
        expected_value = torch.sum(torch.arange(numel))
        self.assertTrue(
            expected_value * (1 - epsilon) <= uniform_sum <=  expected_value * (1 + epsilon),
            msg=f"The expected value is {expected_value}. Your result {uniform_sum} came out too far"
        )

    def test_my_sampler_binomial_multi_dim(self):
        epsilon = 0.002
        numel = 10 ** 7
        binomial_distribution = my_sampler((100, 100, 1000) , [0.5, 0.5])
        binomial_sum = torch.sum(binomial_distribution)
        self.assertTrue(
            numel * 0.5 * (1 - epsilon) <= binomial_sum <=  numel * 0.5 * (1 + epsilon),
            msg=f"The expected value is {numel * 0.5}. Your result {binomial_sum} came out too far"
        )

    def test_grad(self):
        A=my_sampler((2,2),[0.1,0.2,0.7],requires_grad=True)
        B = A * 3
        B[1,0].backward()
        a_grad = torch.tensor([[0, 0],[3, 0]])
        self.assertTrue(torch.equal(A.grad, a_grad), f"expected: {a_grad}. actual: {A.grad}")
    
    def test_no_grad(self):
        A=my_sampler((2,2),[0.1,0.2,0.7])
        B = A * 3
        try:
            B[1,0].backward()
            self.assertTrue(False, "autograd should be off, causing an error to be raised.")
        except RuntimeError as e:
            self.assertEqual(str(e), "element 0 of tensors does not require grad and does not have a grad_fn", msg="unexpected error occured")
        self.assertIsNone(A.grad, f"expected A.grad to be None. actual: {A.grad}")


def run_tests():
    # Run only the tests in the specified classes
    test_classes_to_run = [TestDistValid, TestMySampler]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)

run_tests()
