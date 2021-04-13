clear;clc;

% Load dataset and define features
dataset = 'cleveland_heart_disease_dataset_labelled';
data = struct_data(strcat(dataset,'.mat'));

% Parse the data from the dataset
% Training Data
x = data.training.input';
y = data.training.output';

% Test Data
x_test = data.test.input';
y_test = data.test.output';

% Validation Data
x_val = data.validation.input';
y_val = data.validation.output';

% define input and output features
n_features = data.input_count;
n_output_features = data.output_count;
n_data = data.training_count;                     % Number of samples in training set
n_test_data = data.test_count;                    % Number of samples in test set

% Construct MLP architecture
% Don't need to send e bias terms, 
% which is why the 'bias' option is set to 'false'.
% Bias and Weights will automatically generate in the MLPNet class
net = MLPNet();
net.AddInputLayer(n_features,false);
net.AddHiddenLayer(25,'relu',false);              % 3rd order Network for robust perfomance
net.AddOutputLayer(n_output_features,'softmax',false);

% Intializing the network with learning rate = 0.0005; momentum = adam, 
% loss function = cross entropy & regularization = L1 norm
net.NetParams('rate',0.005,'momentum','adam','lossfun','crossentropy',...
    'regularization','L2');
net.trainable = true;
net.Summary();

% Training parameters
pre_acc = 0;                                      % pre-allocate training accuracy
n_batch = 30;                                     % Size of the minibatch
max_epoch = 500;                                  % Maximum number of epochs
max_batch_index = floor(n_data/n_batch);          % Maximum batch index
max_num_batches = max_batch_index.*max_epoch;     % Maximum number of batches

% Pre-allocate for epoch and error vectors (for max iteration)
epoch = zeros(1,max_num_batches-1);
d_loss = epoch;
ce_test = zeros(max_epoch,1);
ce_train = zeros(max_epoch,1);
ce_val = zeros(max_epoch,1);

% Initialize iterator and timer
batch_index = 1;                                  % Index for  minibatches
epoch_index = 1;                                  % Index for epochs
target_accuracy = 98;                             % Desired classification accuracy

% Checks the max ecpoch condition also checks the pre-allocated...
% accuracy should be less than target accuracy
while ((epoch(batch_index)<max_epoch)&&(pre_acc<target_accuracy))
    
    % Compute current epoch
    epoch(batch_index+1) = batch_index*n_batch/n_data;
    
    % Randomly generate the indices for a minibatch
    rand_ind = randsample(n_data,n_batch);
    
    % Index into input and output data for minibatch
    X_batch = x(rand_ind,:);                       
    Y_batch = y(rand_ind,:);                       
    
    % Train model
    d_loss(batch_index+1) = net.training(X_batch,Y_batch)./n_batch;
    
    % Only compute error/classification metrics after each epoch
    if ~(mod(batch_index,max_batch_index))
        
        % Compute error metrics for training, test, and validation set
        [~,ce_train(epoch_index),~,weights(epoch_index,:), bias(epoch_index,:)]=net.NetworkError(x,y,'classification');
        [~,ce_val(epoch_index),~,weights(epoch_index,:),bias(epoch_index,:)]=net.NetworkError(x_val,y_val,'classification');
        tic;
        [~,ce_test(epoch_index),~,weights(epoch_index,:),bias(epoch_index,:)]=net.NetworkError(x_test,y_test,'classification');
        fprintf('\n-----------End of Epoch: %i------------\n', epoch_index);
        fprintf('Loss function: %f \n',d_loss(batch_index+1));
        fprintf('Test Accuracy: %f Training Accuracy: %f \n',1-ce_test(epoch_index),1-ce_train(epoch_index));
        pre_acc = (1-min(ce_test(epoch_index)));
        epoch_index = epoch_index+1;                    % Update epoch index
    end
    % Update batch index
    batch_index = batch_index+1;
end

% Remove trailing zeros if training met target accuracy before maximum
% number of epochs
ce_test = ce_test(1:(epoch_index-1));
ce_train = ce_train(1:(epoch_index-1));
ce_val = ce_val(1:(epoch_index-1));
fprintf('\n\nMax Test accuracy achieved: %.2f%%\n\n',(1-min(ce_test))*100);
fprintf('Max Train accuracy achieved: %.2f%%\n\n',(1-min(ce_train))*100);

% Plot classification results
figure(1)
plot(movmean((ce_test),3), 'linewidth', 2.5);hold on;
plot(movmean((ce_train),3),'linewidth', 2.5);hold on;
plot(movmean((ce_val),3),'linewidth', 2.5);hold off;
grid on;
ylim([0 1]);
xlabel('Epoch');
ylabel('Error');
legend('Test','Training','Validation');

% Testing the network
max_performance_index = find(ce_test==min(ce_test));

% Retreiving max accuaracy performed indexed weights and bias for testing
max_performance_index = max_performance_index(end,end);

% Predicting the values with retrived weights
y_predict = net.test(x_test, weights(max_performance_index,:), bias(max_performance_index,:));
for i = 1:length(y_predict)
    max_val = max([y_predict(i,1),y_predict(i,2),y_predict(i,3)]);
    if y_predict(i,1) == max_val y_predict(i,1) = 1; else y_predict(i,1) = 0; end
    if y_predict(i,2) == max_val y_predict(i,2) = 1; else y_predict(i,2) = 0; end
    if y_predict(i,3) == max_val y_predict(i,3) = 1; else y_predict(i,3) = 0; end
end

% PLotting confusion matrices of each class
figure(2) 
subplot(2,2,1)
mat1 = confusionchart(y_test(:,1),y_predict(:,1));
title('No heart disease confusion matrix');
subplot(2,2,2)
mat2 = confusionchart(y_test(:,2),y_predict(:,2));
title('Mild heart disease confusion matrix');
subplot(2,2,3)
mat3 = confusionchart(y_test(:,3),y_predict(:,3));
title('Severe heart disease confusion matrix');

