load('cleveland_heart_disease_dataset_labelled.mat')
data_set = [x t];%Create the data set in matrix form
col_names ={'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope','ca', 'thal', 'num'};%Column labels for the input data
data = array2table(data_set, 'VariableNames', col_names); %Append the column name to the table

%Data visualization with age,disease count and classification type
x1 = unique(data.age); % gives the unique value of the age 
dis_0 = data((data.num == 0),:); %Return the rows of dataset with target value 0
dis_1 = data((data.num == 1),:); %Return the rows of dataset with target value 1
dis_2 = data((data.num == 2),:); %Return the rows of dataset with target value 2

y1=zeros(41,3); % form a intial matrix for storing age count based on target values
for i = 1:length(x1)%Loop for getting target count values with respect to all ages
    a=size((dis_0((dis_0.age == x1(i,:)),1)),1);
    b=size((dis_1((dis_1.age == x1(i,:)),1)),1); 
    c=size((dis_2((dis_2.age == x1(i,:)),1)),1);
    y1(i,:) = [a, b, c];
end
figure(1)
subplot(2,2,1)
bar(x1,y1,1.3)%bar graph of the classified data
legend("No disease","Mild Heart disease","Severe heart disease")
title("Age vs Heart disease")
xlabel("Age")
ylabel("count")

%PIE CHART of MALE WITH DIFFERENT HEART DISEASE
M0=size((dis_0((dis_0.sex == 1),2)),1);
M1=size((dis_1((dis_1.sex == 1),2)),1);
M2=size((dis_2((dis_2.sex == 1),2)),1);
male = [M0 M1 M2];
subplot(2,2,2)
pie(male)
legend("No disease","Mild Heart disease","Severe heart disease")
title("Male with Heart Disease")

%PIE CHART OF FEMALE WITH DIFFERENT HEART DISEASE
F0=size((dis_0((dis_0.sex == 0),2)),1);
F1=size((dis_1((dis_1.sex == 0),2)),1);
F2=size((dis_2((dis_2.sex == 0),2)),1);
female = [F0 F1 F2];
subplot(2,2,3)
pie(female)
legend("No disease","Mild Heart disease","Severe heart disease")
title("Female with Heart Disease")

%CHEST PAIN CLASSIFICATION W.R.T TARGET
y2=zeros(4,3); % form a intial matrix for storing age count based on target values
for i = 1:4%Loop for getting target count values with respect to all ages
    a=size(data((data.num == 0)&(data.cp == i),:),1);
    b=size(data((data.num == 1)&(data.cp == i),:),1);
    c=size(data((data.num == 2)&(data.cp == i),:),1);
    y2(i,:) = [a, b, c];
end
x2 = categorical({'typical angina', 'atypical angina','non-anginal pain', 'asymptomatic'});
subplot(2,2,4)
bar(x2,y2)%bar graph of the classified data
legend("No disease","Mild Heart disease","Severe heart disease")
title("Chest pain vs Heart disease")
xlabel("Chest pain")
ylabel("count")
