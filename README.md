# OUI-Deep-Learning-Tests
Tests for the Deep Learning course assignments in the Open University of Israel


# Instructions for running

Copy and paste the test as a code block at the end of the jupyter notebook for the relevant exercise. 

Set the environment code block to match the function names that you chose:
For example, if you chose to call your "expand_as" function "nyancat", set the variable to match:
```python3
environment = {
    "expand_as_function": nyancat,
    "broadcastable_together_function": broadcastable_together,
    "broadcast_tensors_function": broadcast_tensors,
    "verbose": True
}
```
The "verbose" flag lets you see the errors causing a test to fail.

Finally run the code block (make sure you ran the rest of the notebook first)

# Note

Some tests may be too stringent, so don't worry if you see failures - they might be because of things like returning a list instead of a tuple.
Contributions are welcome! Contact me if you want to contribute
