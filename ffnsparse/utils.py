

class EarlyStopper:
    def __init__(self, patience=1, threshold=1e-4):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        ratio_increase = ((validation_loss - self.min_validation_loss) / 
                                      self.min_validation_loss)
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif ratio_increase > self.threshold:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False