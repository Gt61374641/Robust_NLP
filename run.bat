@echo off
REM ========================================
REM UniGen Pipeline Runner
REM Usage: run.bat [step]
REM Steps: generate, solve, train, test
REM ========================================

SET GEN_MODEL=gpt2_xl
SET GEN_PROMPT=gen_uni_p1
SET CLS_PROMPT=cls_p1
SET GEN_AMOUNT=1000
SET TOP_K=40
SET TOP_P=0.9
SET TEMP=1.0
SET RE_TEMP=0.1
SET RE_TH=0.2
SET TAM_MODEL=roberta
SET DATASET=sst2
SET BATCH=32
SET EPOCHS=3
SET DEVICE=cuda

REM Common generation params (must match for file naming)
SET GEN_PARAMS=--gen_model_type %GEN_MODEL% --gen_prompt %GEN_PROMPT% --gen_amount %GEN_AMOUNT% --gen_top_k %TOP_K% --gen_top_p %TOP_P% --gen_temperature %TEMP% --gen_relabel_temperature %RE_TEMP% --gen_relabel_threshold %RE_TH%
SET COMMON=--task classification --num_workers 0 --device %DEVICE% --use_wandb False

if "%1"=="generate" (
    echo === Step 1: Generating data with %GEN_MODEL% ===
    python main.py --job generating --model_type %GEN_MODEL% --generation_type unigen --gen_relabel soft --batch_size 8 %GEN_PARAMS% %COMMON%
    goto :end
)

if "%1"=="solve" (
    echo === Step 2: SunGen Solve ===
    python main.py --job sungen_solve --training_type unigen --model_type %TAM_MODEL% --task_dataset %DATASET% --cls_prompt %CLS_PROMPT% --batch_size %BATCH% --sungen_outer_epoch 10 --sungen_valid_size 100 --sungen_train_size 800 %GEN_PARAMS% %COMMON%
    goto :end
)

if "%1"=="train" (
    echo === Step 3: Training %TAM_MODEL% TAM ===
    python main.py --job training --training_type unigen --model_type %TAM_MODEL% --task_dataset %DATASET% --cls_prompt %CLS_PROMPT% --batch_size %BATCH% --num_epochs %EPOCHS% %GEN_PARAMS% %COMMON%
    goto :end
)

if "%1"=="test" (
    echo === Step 4: Testing on all datasets ===
    for %%d in (sst2 imdb rotten cr yelp_polarity amazon_polarity financial_phrasebank) do (
        echo --- Testing on %%d ---
        python main.py --job testing --training_type unigen --model_type %TAM_MODEL% --task_dataset %DATASET% --test_dataset %%d --cls_prompt %CLS_PROMPT% --batch_size %BATCH% %GEN_PARAMS% %COMMON%
    )
    goto :end
)

echo Usage: run.bat [generate / solve / train / test]
echo.
echo Example:
echo   run.bat generate    - Generate synthetic data
echo   run.bat solve       - Run SunGen bilevel optimization
echo   run.bat train       - Train TAM model
echo   run.bat test        - Test on all 7 datasets

:end
