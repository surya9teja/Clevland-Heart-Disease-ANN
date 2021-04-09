clear all;clc;
% Load dataset and define features
dataset = 'cleveland_heart_disease_dataset_labelled';      % Choose between MNIST and Fashion_MNIST
data = struct_data(strcat(dataset,'.mat'));
% Parse the data from the dataset
% Training Data
X = data.training.input';
Y = data.training.output';
% Test Data
X_test = data.test.input';
Y_test = data.test.output';
% Validation Data
X_val = data.validation.input';
Y_val = data.validation.output';
% define input and output features
n_features = data.input_count;
n_output_features = data.output_count;
n_data = data.training_count;   % Number of samples in training set
n_test_data = data.test_count;  % Number of samples in test set
% Construct MLP architecture
% The included datasets already have bias terms, which is why the 'bias' option is set
% to 'false'.
network = MLPNet();
network.AddInputLayer(n_features,false);
network.AddHiddenLayer(22,'leakyrelu',false);
network.AddHiddenLayer(11,'leakyrelu',false);
network.AddOutputLayer(n_output_features,'softmax',false);
network.NetParams('rate',0.0005,'momentum','adam','lossfun','crossentropy',...
    'regularization','L2');
network.trainable = true;
network.Summary();
% Training parameters
acc = 0;                        % pre-allocate training accuracy
n_batch = 50;                  % Size of the minibatch
max_epoch = 600;                 % Maximum number of epochs
max_batch_idx = floor(n_data/n_batch);          % Maximum batch index
max_num_batches = max_batch_idx.*max_epoch;     % Maximum number of batches
% Pre-allocate for epoch and error vectors (for max iteration)
epoch = zeros(1,max_num_batches-1);
d_loss = epoch;
ce_test = zeros(max_epoch,1);
ce_train = zeros(max_epoch,1);
ce_val = zeros(max_epoch,1);
% Initialize iterator and timer
batch_idx = 1;      % Index to keep track of minibatches
epoch_idx = 1;      % Index to keep track of epochs
target_accuracy = 98; % Desired classification accuracy
while ((epoch(batch_idx)<max_epoch)&&(acc<target_accuracy))
    
    % Compute current epoch
    epoch(batch_idx+1) = batch_idx*n_batch/n_data;
    % Randomly sample data to create a minibatch
    rand_ind = randsample(n_data,n_batch);
    % Index into input and output data for minibatch
    X_batch = X(rand_ind,:);    % Sample Input layer
    Y_batch = Y(rand_ind,:);    % Sample Output layer
    
    % Train model
    d_loss(batch_idx+1) = network.training(X_batch,Y_batch)./n_batch;
    
    % Only compute error/classification metrics after each epoch
    if ~(mod(batch_idx,max_batch_idx))
        % Compute error metrics for training, test, and validation set
        [~,ce_train(epoch_idx),~]=network.NetworkError(X,Y,'classification');
        [~,ce_val(epoch_idx),~]=network.NetworkError(X_val,Y_val,'classification');
        tic;
        [~,ce_test(epoch_idx),~]=network.NetworkError(X_test,Y_test,'classification');
        eval_time = toc;
        fprintf('\n-----------End of Epoch %i------------\n', epoch_idx);
        fprintf('Loss function: %f \n',d_loss(batch_idx+1));
        fprintf('Test Set Accuracy: %f Training Set Accuracy: %f \n',1-ce_test(epoch_idx),1-ce_train(epoch_idx));
        fprintf('Test Set Evaluation Time: %f s\n\n',eval_time);
        acc = (1-ce_test(epoch_idx));
        epoch_idx = epoch_idx+1;    % Update epoch index
    end
    % Update batch index
    batch_idx = batch_idx+1;
end
% Remove trailing zeros if training met target accuracy before maximum
% number of epochs
ce_test = ce_test(1:(epoch_idx-1));
ce_train = ce_train(1:(epoch_idx-1));
ce_val = ce_val(1:(epoch_idx-1));
% Plot classification results
figure(1)
plot(ce_test);hold on;
plot(ce_train);hold on;
plot(ce_val);hold off;
grid on;
xlabel('Epoch');
ylabel('Error');
legend('Test Set','Training Set','Validation Set');
