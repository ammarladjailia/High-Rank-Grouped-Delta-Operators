import math

class DivergenceMonitor:
    def __init__(self, loss_threshold=100.0, patience=3):
        self.loss_threshold = loss_threshold
        self.patience = patience
        
        self.consecutive_high_loss = 0
        self.first_high_proxy_epoch = None
        self.first_sustained_increase_epoch = None
        self.first_threshold_epoch = None
        self.divergence_epoch = None
        self.divergence_reason = None
        
        self.prev_loss = None
        self.is_diverged = False

    def check(self, epoch, avg_train_loss, max_active_proxy):
        if self.is_diverged:
            return True
            
        if max_active_proxy > 1.0 and self.first_high_proxy_epoch is None:
            self.first_high_proxy_epoch = epoch
            
        if self.prev_loss is not None and avg_train_loss > self.prev_loss:
            if self.first_sustained_increase_epoch is None:
                self.first_sustained_increase_epoch = epoch
        else:
            self.first_sustained_increase_epoch = None
            
        self.prev_loss = avg_train_loss
        
        if math.isnan(avg_train_loss) or math.isinf(avg_train_loss):
            self.divergence_epoch = epoch
            self.divergence_reason = "NaN or Inf loss"
            self.is_diverged = True
            return True
            
        if avg_train_loss > self.loss_threshold:
            if self.first_threshold_epoch is None:
                self.first_threshold_epoch = epoch
            self.consecutive_high_loss += 1
            if self.consecutive_high_loss >= self.patience:
                self.divergence_epoch = epoch
                self.divergence_reason = f"Loss > {self.loss_threshold} for {self.patience} epochs"
                self.is_diverged = True
                return True
        else:
            self.consecutive_high_loss = 0
            
        return False
