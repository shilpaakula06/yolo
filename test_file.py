import nbformat
import pytest
import numpy as np
from cnnb.XORNN import *

class TestModel:
    @pytest.fixture(autouse=True)
    def get_model(self):
        x = sigmoid()
        self.x = str(x)
        dx = sigmoid_derivative()
        self.dx = str(dx)
        _, _, _, _, fp1, fp2, fp3, fp4 = forward_prop()
        self.fp1 = str(fp1)
        self.fp2 = str(fp2)
        self.fp3 = str(fp3)
        self.fp4 = str(fp4)
        a1 = np.array([0.5, 0.5, 0.5, 0.5])
        _, _, bp1, bp2 = backword_prop(a1)
        self.bp1 = str(bp1)
        self.bp2 = str(bp2)


    def test_nn(self):
        assert self.x == '0.5'
        assert self.dx == '0.25'
        assert self.fp1 == '(1, 5)'
        assert self.fp2 == '(1, 5)'
        assert self.fp3 == '(1, 5)'
        assert self.fp4 == '(1, 5)'
        assert self.bp1 == '(1,)'
        assert self.bp2 == '(2, 5)'
