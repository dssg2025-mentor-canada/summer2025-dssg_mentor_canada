# Exploring the Impact of Youth Mentorship through Data Science

***Mentor Canada*** advocates for equitable access to high-quality mentorship for youth across Canada, acknowledging that mentors can have a lasting positive impact in helping young people build stronger future-planning skills and achieve better career outcomes. However, significant barriers such as lower socioeconomic status (SES), often limit mentorship access. By leveraging a rich survey dataset comprising over 2,800 responses using data science techniques, we aim to uncover hidden patterns and inequities to catalyze better mentorship practice and public awareness of mentorship. This unique opportunity will deepen the current understanding of youth mentorship’s impact and drive meaning change through the lens of data science, allowing us to translate data-driven insights into actionable recommendations for families, educators, and policymakers.

Supervised by the [**Data Science for Social Good (DSSG)**](https://dsi.ubc.ca/data-science-social-good/2025) fellowhsip program at UBC's Data Science Institute, we use the this repository to share all weekly progress and workflows for our Mentor Canada project.

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