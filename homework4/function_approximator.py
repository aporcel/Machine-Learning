# partially based on code from http://deeplearning.net/tutorial/
import os
import sys
import numpy
import theano
import theano.tensor as T

from mlp import MLP

class FunctionApproximator(object):
    def __init__(self, n_in=1, n_out=1, n_hidden=10): 

        self.n_out = n_out
        self.n_in = n_in
        self.n_hidden = n_hidden

    def train(self, X, Y, learning_rate=0.1, n_epochs=100, report_frequency=10, lambda_l2=0.0):

        self.report_frequency = report_frequency 

        # allocate symbolic variables for the data
        x = T.matrix('x')  
        y = T.vector('y')  

        # put the data in shared memory
        self.shared_x = theano.shared(numpy.asarray(X, dtype=theano.config.floatX))
        self.shared_y = theano.shared(numpy.asarray(Y, dtype=theano.config.floatX))
        rng = numpy.random.RandomState(1234)

        # initialize the mlp
        self.mlp = MLP(rng=rng, input=x, n_in=self.n_in, n_out=self.n_out,
                  n_hidden=self.n_hidden)

        # define the cost function, possibly with regularizing term
        if lambda_l2>0.0:
            cost = self.mlp.cost(y) + lambda_l2*self.mlp.l2
        else:
            cost = self.mlp.cost(y) 

        # compute the gradient of cost with respect to theta (stored in params)
        # the resulting gradients will be stored in a list gparams
        gparams = [T.grad(cost, param) for param in self.mlp.params]

        updates = [(param, param - learning_rate * gparam)
            for param, gparam in zip(self.mlp.params, gparams) ]

        # compiling a Theano function `train_model` that returns the cost, but
        # at the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(
            inputs=[],
            outputs=cost,
            updates=updates,
            givens={
                x: self.shared_x,
                y: self.shared_y
            }
        )

        #define function that returns model prediction
        self.predict_model = theano.function(
            inputs=[self.mlp.input], outputs=self.mlp.y_pred)

        ###############
        # TRAIN MODEL #
        ###############

        epoch = 0

        while (epoch < n_epochs):
            epoch = epoch + 1
            epoch_cost = train_model()
            if epoch % self.report_frequency == 0:
                print("epoch: %d  cost: %f" % (epoch, epoch_cost))

    def get_y_pred(self, x=None):
        if x is None:
            return self.predict_model(self.shared_x.get_value())    
        else:
            return self.predict_model(x)

    def get_weights(self):
        return [self.mlp.params[0].get_value(), self.mlp.params[1].get_value(), self.mlp.params[2].get_value(), self.mlp.params[3].get_value()]
