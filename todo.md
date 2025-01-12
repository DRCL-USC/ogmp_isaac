## Todo before release 

- [x] Add list mode for generate.py
- [x] move thresh, start target into the box env computed based on box dim 
- [ ] Add the last policy of recreated experiments for play.py
- [ ] Setup the base.yml and vary.yaml for each task in `exts/ogmp_isaac/config` for recreating trainings. 
- [ ] Remove unused robot models
- [ ] Documentation 
    - [ ] Installation (version notes) 
    - [ ] play.py, train.py
    - [ ] generate.py, deploy.py
    - [ ] citation and acknowledgement

## Results Recreated in Refactored Code

| Task\Robot | HECTOR_V1P5 |  BERKELEY_HUMANOID | G1 | H1 | 
| ---------- | ----------- |  -----------        | ----------- | ----------- |
| Soccer w/ kick| | :white_check_mark: |:white_check_mark: |:white_check_mark: |    
| Box push|:white_check_mark: | :white_check_mark: |||   