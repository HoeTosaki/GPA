import matplotlib.pyplot as plt

print('dwad')

if __name__ == '__main__':
    plt.scatter(range(100),[1]*100,c=[ele/100 for ele in range(100)])
    plt.show()