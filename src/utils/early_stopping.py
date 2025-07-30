class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, counter=0, metric_type="ascending"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = counter
        self.metric_type = metric_type
        # The best_metric value is initialized based on the metric_type
        self.best_metric = float('-inf') if metric_type == "ascending" else float('inf')
        self.early_stop = False

    def __call__(self, metric):
        is_better = False
        if self.metric_type == "ascending":
            if metric > self.best_metric + self.min_delta:
                is_better = True
        else:  # descending
            if metric < self.best_metric - self.min_delta:
                is_better = True

        if is_better:
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                self.early_stop = True

    def state_dict(self):
        """
        Returns the state of the EarlyStopping instance.
        """
        return {
            'patience': self.patience,
            'min_delta': self.min_delta,
            'counter': self.counter,
            'metric_type': self.metric_type,
            'best_metric': self.best_metric,
            'early_stop': self.early_stop,
        }

    def load_state_dict(self, state_dict):
        """
        Loads the EarlyStopping state.
        """
        self.patience = state_dict['patience']
        self.min_delta = state_dict['min_delta']
        self.counter = state_dict['counter']
        self.metric_type = state_dict['metric_type']
        self.best_metric = state_dict['best_metric']
        self.early_stop = state_dict['early_stop']