current_test_case = 0


pass_count = 0
fail_count = 0

code_success = 0
code_bad_error_handling = 1
code_bad_result = 2
code_bad_result_broadcastable_together = 3
code_bad_return_type = 4
code_incorrect_output_length = 5
environment = {
    "expand_as_function": expand_as,
    "broadcastable_together_function": broadcastable_together,
    "broadcast_tensors_function": broadcast_tensors,
    "verbose": True
}

def print_pass_and_increment_test_case_counter():
    global current_test_case, pass_count
    pass_count += 1
    print("=================================")
    print(f"PASS #{current_test_case}")
    current_test_case += 1

def print_fail_and_increment_test_case_counter(expected, actual, code):
    global current_test_case, fail_count
    fail_count += 1
    print("=================================")
    print(f"FAIL #{current_test_case}")
    if environment["verbose"]:
        if code == code_bad_error_handling:
            print("expected an error to be thrown") if expected else print("did not expect an error to be thrown")
        elif code == code_bad_result:
            print(f"expected: {expected.head()}", f"actual: {actual.head()}")
        elif code == code_bad_result_broadcastable_together:
            print(f"expected: {expected}", f"actual: {actual}")
        elif code == code_bad_return_type:
            print(f"bad return type: {type(actual)}, expected: {type(expected)}")
        elif code == code_incorrect_output_length:
            print(f"bad output length: {len(actual)}. expected: {len(expected)}")
    current_test_case += 1
    
def compare_expand_as(A, B):
    expected_error, threw_error = False, False
    try:
        expected = A.expand_as(B)
    except:
        expected_error = True
    try:
        actual = environment["expand_as_function"](A, B)
    except:
        threw_error = True
    if expected_error != threw_error:
        return (code_bad_error_handling, (expected_error, threw_error))
    if threw_error:
        return (code_success, None)
    if not torch.equal(expected, actual):
        return (code_bad_result, (expected, actual))
    return (code_success, None)


def compare_broadcastable_together(A, B):
    expected = False
    try:
        expected = True, torch.broadcast_tensors(A, B)[0].shape
    except:
        pass
    try:
        actual = environment["broadcastable_together_function"](A, B)
    except:
        return (code_bad_error_handling, (False, True))
    if type(expected) != type(actual):
        return (code_bad_result_broadcastable_together, (expected, actual))
    elif type(expected) == tuple == type(actual) and (expected[0] != actual[0] or expected[1] != actual[1]):
        return (code_bad_result_broadcastable_together, (expected, actual))
    elif type(expected) == bool == type(actual) and expected != actual:
        return (code_bad_result_broadcastable_together, (expected, actual))
    return (code_success, None)


def compare_broadcast_tensors(A, B):
    expected_error, threw_error = False, False
    try:
        expected = torch.broadcast_tensors(A, B)
    except:
        expected_error = True
    try:
        actual = environment["broadcast_tensors_function"](A, B)
    except:
        threw_error = True
    if expected_error != threw_error:
        return (code_bad_error_handling, (expected_error, threw_error))
    if threw_error:
        return (code_success, None)
    if type(actual) != type(expected):
        return (code_bad_return_type, (expected, actual))
    if len(expected) != 2:
        return (code_incorrect_output_length, (expected, actual))
    expected_a, expected_b = expected
    actual_a, actual_b = actual
    if not torch.equal(expected_a, actual_a):
        print("bad result for a")
        return (code_bad_result, (expected_a, actual_a))
    if not torch.equal(expected_b, actual_b):
        print("bad result for b")
        return (code_bad_result, (expected_b, actual_b))
    return (code_success, None)




test_cases = [
    [torch.tensor([3, 3]), torch.tensor([2])],
    [torch.tensor([[1,3,5],[1,3,5]]), torch.tensor([2])],
    [torch.tensor([1,2]), torch.tensor([[2,3,4], [5,6,7]])],
    [torch.tensor([1,2,3]), torch.tensor([[2,3,4], [5,6,7]])],
    [torch.tensor([[1,2,3]]), torch.tensor([[2,3,4], [5,6,7]])],
    
    [torch.tensor([[[1,2,3]]]), torch.tensor([[2,3,4], [5,6,7]])],
    [torch.arange(10**4).reshape(10, 10, 10, 1, 10), torch.arange(10**5).view(10, 10, 10, 10, 10)],
    [torch.arange(10**3).reshape(10, 1, 10, 1, 10), torch.arange(10**5).view(10, 10, 10, 10, 10)],
    [torch.arange(10**3).reshape(10, 10, 1, 10), torch.arange(10**5).view(10, 10, 10, 10, 10)],
    [torch.arange(10**2).reshape(10, 1, 1, 10), torch.arange(10**5).view(10, 10, 10, 10, 10)],
    
    [torch.arange(10**2).reshape(10, 10), torch.arange(10**5).view(10, 10, 10, 10, 10)]
]




def run_suite(compare_fn):
    global current_test_case
    for i, AB in enumerate(test_cases):
        status_code, result = compare_fn(*AB)
        if status_code != code_success:
            print_fail_and_increment_test_case_counter(*result, status_code)
            continue
        print_pass_and_increment_test_case_counter()
    print("==================================================================")
    print("REVERSING INPUTS")
    current_test_case = 0
    for i, AB in enumerate(test_cases):
        status_code, result = compare_fn(*AB[::-1])
        if status_code != code_success:
            print_fail_and_increment_test_case_counter(*result, status_code)
            continue
        print_pass_and_increment_test_case_counter()
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    
print("Starting expand_as tests")
run_suite(compare_expand_as)
print("Starting broadcastable_together tests")
run_suite(compare_broadcastable_together)
print("Starting broadcast_tensors tests")
run_suite(compare_broadcast_tensors)



if fail_count == 0:
    print(f"PASSED ALL {pass_count} TESTS!")
else:
    print(f"PASSED: {pass_count} FAILED: {fail_count}")
