for i = 1:12
    temp = csvread(strcat("data/shield_test_", num2str(i),".dat"));
    A(:,i) = temp(:,4);
end
A = mean(A,2);
plot(A)
title("Average reward during training, 12 samples")
xlabel("Learning episode")
ylabel("Reward")
xlim([0 1000])

for i = 1:12
    temp = csvread(strcat("data/mc_shield_test_", num2str(i),".dat"));
    A(:,i) = temp(:,4);
end
A = mean(A,2);
plot(A)
title("Average reward during training, 12 samples")
xlabel("Learning episode")
ylabel("Reward")
xlim([0 1000])

