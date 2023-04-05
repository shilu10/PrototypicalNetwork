class PrototypicalModel:
    
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def train(self, train_x, train_y, n_way, n_support, 
                                  n_query, max_epoch, epoch_size):
        """
          Trains the protonet
          Args:
              model
              optimizer
              train_x (np.array): images of training set
              train_y(np.array): labels of training set
              n_way (int): number of classes in a classification task
              n_support (int): number of labeled examples per class in the support set
              n_query (int): number of labeled examples per class in the query set
              max_epoch (int): max epochs to train on
              epoch_size (int): episodes per epoch
        """
      #divide the learning rate by 2 at each epoch, as suggested in paper
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
        epoch = 0 #epochs done so far
        stop = False #status to know when to stop

        while epoch < max_epoch and not stop:
            running_loss = 0.0
            running_acc = 0.0

            for episode in tnrange(epoch_size, desc="Epoch {:d} train".format(epoch+1)):
                sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
                self.optimizer.zero_grad()
                loss, output = self.model.set_forward_loss(sample)
                running_loss += output['loss']
                running_acc += output['acc']
                loss.backward()
                optimizer.step()
            epoch_loss = running_loss / epoch_size
            epoch_acc = running_acc / epoch_size
            print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))
            epoch += 1
            scheduler.step()
            
        return self.model
    
    def test(self, trained_model, test_x, test_y, test_episode): 
        """
          Tests the protonet
          Args:
              model: trained model
              test_x (np.array): images of testing set
              test_y (np.array): labels of testing set
              n_way (int): number of classes in a classification task
              n_support (int): number of labeled examples per class in the support set
              n_query (int): number of labeled examples per class in the query set
              test_episode (int): number of episodes to test on
        """
        running_loss = 0.0
        running_acc = 0.0
        for episode in tnrange(test_episode):
            sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
            loss, output = trained_model.set_forward_loss(sample)
            running_loss += output['loss']
            running_acc += output['acc']
        avg_loss = running_loss / test_episode
        avg_acc = running_acc / test_episode
        print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc)) 
