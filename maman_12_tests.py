import unittest

environment = {
    "function_dist_valid":  is_distribution_valid,  #  input: parameter dist of my_sampler. output: True iff distribution is valid else False
    "q2_lib": {
        "add": myadd,
        "multiply": mymulti,
        "power": power,
        "cos": cosemet,
        "sin": sinusitis,
        "ln": ln,
        "exp": exp
    },
    "get_gradient_descending_insertion": False,

    "test_dist_valid": True,
    "test_my_sampler": True,
    "test_from_book": True,
    "test_all_ops": True
}

q2_lib = environment["q2_lib"]

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

    def test_my_sampler_grad(self):
        A=my_sampler((2,2),[0.1,0.2,0.7],requires_grad=True)
        B = A * 3
        B[1,0].backward()
        a_grad = torch.tensor([[0, 0],[3, 0]])
        self.assertTrue(torch.equal(A.grad, a_grad), f"expected: {a_grad}. actual: {A.grad}")
    
    def test_my_sampler_no_grad(self):
        A=my_sampler((2,2),[0.1,0.2,0.7])
        B = A * 3
        try:
            B[1,0].backward()
            self.assertTrue(False, "autograd should be off, causing an error to be raised.")
        except RuntimeError as e:
            self.assertEqual(str(e), "element 0 of tensors does not require grad and does not have a grad_fn", msg="unexpected error occured")
        self.assertIsNone(A.grad, f"expected A.grad to be None. actual: {A.grad}")


class TestGrad(unittest.TestCase):
    @unittest.skipIf(reason="chose to skip", condition=(environment["test_from_book"] == False))
    def test_from_book(self):
        a = torch.tensor([2], dtype=float, requires_grad=True)
        b = a ** 2
        b.retain_grad()
        c = torch.exp(b)
        c.retain_grad()
        c.backward()
        grads = [c.grad.item(), b.grad.item(), a.grad.item()]
        expected = [round(grad, 3) for grad in grads]
        if environment["get_gradient_descending_insertion"]:
            expected = expected[::-1]

        a=MyScalar(2)
        b=q2_lib["power"](a, 2)
        c=q2_lib["exp"](b)
        d=get_gradient(c)
        actual = [round(grad, 3) for grad in d.values()]
        self.assertEqual(expected, actual, f'if the values are in reverse order set "get_gradient_descending_insertion": True in the environment')
    @unittest.skipIf(reason="chose to skip", condition=(environment["test_all_ops"] == False))
    def test_all_ops(self):
        n2 = 0.3
        
        a = torch.tensor([3], dtype=float, requires_grad=True)
        
        b = a + 3
        b.retain_grad()
        
        c = n2 * b
        c.retain_grad()
        
        d = c ** 5
        d.retain_grad()

        e = torch.exp(d)
        e.retain_grad()

        f = torch.log(e)
        f.retain_grad()

        g = torch.cos(f)
        g.retain_grad()

        h = torch.sin(g)
        h.retain_grad()

        h.backward()
        grads = [x.grad.item() for x in (h,g,f,e,d,c,b,a)]

        expected = [round(grad, 3) for grad in grads]
        if environment["get_gradient_descending_insertion"]:
            expected = expected[::-1]

        a=MyScalar(3)
        print(a)
        b = q2_lib["add"](a, 3)
        print(b)
        c = q2_lib["multiply"](b, n2)
        print(c)
        d = q2_lib["power"](c, 5)
        print(d)
        e = q2_lib["exp"](d)
        print(e)
        f = q2_lib["ln"](e)
        g = q2_lib["cos"](f)
        h = q2_lib["sin"](g)
        i = get_gradient(h)
        actual = [round(grad, 3) for grad in i.values()]
        self.assertEqual(expected, actual, f'if the values are in reverse order set "get_gradient_descending_insertion": True in the environment')

def run_tests():
    # Run only the tests in the specified classes
    test_classes_to_run = [TestDistValid, TestMySampler, TestGrad]

    loader = unittest.TestLoader()

    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
        
    big_suite = unittest.TestSuite(suites_list)

    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)

run_tests()
