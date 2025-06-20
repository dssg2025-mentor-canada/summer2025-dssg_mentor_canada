# summer2025-dssg_mentor_canada

## To replicate this analysis' python environment:

-   To see all python package dependencies, you can review the [`environment.yml`](https://github.com/dssg2025-mentor-canada/summer2025-dssg_mentor_canada/blob/main/environment.yml) file for our `dssg_env` environment.

-   If you have **newly cloned** down the current repository and looking to initialize this replicable `dssg_env` environment, please copy and run the following command lines in your **terminal application**:

    1.  `conda env create -f environment.yml`

    2.  `conda activate dssg_env`

    > To check for all existing environments on your local machine, you can run the following command:
    >
    > ```         
    > conda env list
    > ```

-   If you already have a `dssg_env` environment on your local machine and wish to update it due to added dependencies in [`environment.yml`](https://github.com/dssg2025-mentor-canada/summer2025-dssg_mentor_canada/blob/main/environment.yml), you can follow the subsequent steps:

    1.  Pull the updated environment.yml

    2.  Open your terminal application and run the following command:

        ```         
        conda env update -f environment.yml --name dssg_env
        ```

    3.  Patiently wait for package downloads & then run the following command to activate `dssg_env`:

        ```         
        conda activate dssg_env 
        ```

    4.  Good to go!