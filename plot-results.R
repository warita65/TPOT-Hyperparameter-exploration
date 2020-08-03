
library(tidyverse)
knn_res <- read_csv('https://raw.githubusercontent.com/warita65/TPOT-Hyperparameter-exploration/master/tpot-KNN_(range(1%2Cn_observations_2)%20output%20table%20-%20Sheet1.csv',
                    skip = 1) %>%
  filter(`KNN config for n_neighbors` == 'range(1,n_observations/2),')

knn_def <- read_csv('https://raw.githubusercontent.com/warita65/TPOT-Hyperparameter-exploration/master/tpot-KNN%20default%20values%20output%20table%20-%20Sheet2.csv') %>% 
  select(-c(X8, X13, X18)) %>% 
  rename(nneighbors_1 = `n_neighbors ran#1`,
         nneighbors_2 = `n_neighbors ran#2`,
         nneighbors_3 = `n_neighbors ran#3`,
         nneighbors_4 = `n_neighbors ran#4`,
         p_1 = p,
         p_2 = p2,
         p_3 = p3,
         p_4 = p4,
         weight_1 = weight,
         weight_2 = weights2,
         weight_3 = weights3,
         weight_4 = weights4,
         score_1 = score1,
         score_2 = score2,
         score_3 = scorer3,
         score_4 = score4,
         n_obs = `# instances`,
         n_feats = `# features`,
         dataset = `Dataset name`)


pivoted_knn <- knn_def %>%
  group_by(dataset, n_obs, n_feats) %>% 
  pivot_longer(cols = matches('nneighbors|weight|p_|score'),
               names_to = c(".value", "set"),
               names_sep = '_'
  )

pivoted_knn_small <- pivoted_knn %>% 
  filter(set == 1, n_feats < 1000) 

pivoted_knn_small %>%
  ggplot(aes(x = n_feats, y = nneighbors)) +
  geom_point() +
  geom_smooth(method = 'lm')


pivoted_knn_small %>% 
  ggplot(aes(x = n_obs, y = nneighbors)) +
  geom_point() +
  geom_smooth(method = 'lm')

cor.test(pivoted_knn_small$n_obs, pivoted_knn_small$nneighbors)
cor.test(pivoted_knn_small$n_feats, pivoted_knn_small$nneighbors)




