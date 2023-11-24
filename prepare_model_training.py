def prepare_data(data, label):
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(data.values, label.reshape(-1,1), test_size=0.2, random_state=2023)
    return train_X, test_X, train_y, test_y