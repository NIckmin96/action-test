from load_iris import *

if __name__ == '__main__':
    model_pipeline(*prepare_data(*load_data()))