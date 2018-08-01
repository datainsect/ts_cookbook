import numpy as np
import tensorflow as tf

def generateLinearData(dimension, num):
    """
    """
    np.random.seed(1024)
    beta = np.array(range(dimension)) + 1
    x = np.random.random((num, dimension))
    epsilon = np.random.random((num, 1))

    y = x.dot(beta).reshape((-1, 1)) + epsilon
    return x, y


def createLinearModel(dimenssion):
    """
    dimenssion:num of independent variables
    model: dict the linear model
    """
    x = tf.placeholder(tf.float64,[None,dimenssion],name='x')
    y = tf.placeholder(tf.float64,[None,1],name='y')
    betaPred = tf.Variable(np.random.random([dimenssion,1]))
    y_pred = tf.matmul(x,betaPred,name="y_pred")
    loss = tf.reduce_mean(tf.square(y-y_pred))
    model = {'loss':loss,'x':x,'y':y,'paras':betaPred,"y_pred":y_pred}
    return model

def gradientDescent(X,Y,model,learningRate=0.01,maxIter=10000,tol=1.e-1):
    """
    """
    method = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
    optimizer = method.minimize(model['loss'])
    sess = tf.Session()
    init =tf.global_variables_initializer()
    sess.run(init)
    step =0 
    diff = np.inf
    pre_loss = np.inf
    while step<maxIter and diff>tol:
        _,loss = sess.run(
            [optimizer,model['loss']],
            feed_dict = {model['x']:X,model['y']:Y}
        )
        diff = pre_loss - loss
        pre_loss = loss
        step += 1
        if step%25==0 :
            print('loss:{0}\tdiff:{1}'.format(loss,diff))
    # print(model['loss'].eval(session=sess))

if __name__ == '__main__':
    dimension = 30
    num = 10000
    X,Y = generateLinearData(dimension,num)
    model = createLinearModel(dimension)
    gradientDescent(X,Y,model)