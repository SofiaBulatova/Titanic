from Config import config
from train import train
from Prediction import test

def main():
    train(config)
    test(config)

# Запуск пайплайна
if __name__ == '__main__':
    main()