from typing import Optional
import pandas as pd

def get_first_failed_tests(solutions_df: pd.DataFrame, tests_df:  pd.DataFrame) -> pd.Series:    
    # Merge solutions with tests based on task_id
    merged_df = pd.merge(solutions_df, tests_df, on="task_id", how="inner", validate="one_to_many")
    
    # Group by 'id' (submission_id) and select the first test input for each submission
    first_failed_tests = merged_df.groupby('id').first()['input']
    
    return first_failed_tests

import pandas as pd
from typing import Optional

def get_error_messages(student_code: str, solutions_df: pd.DataFrame, tests_df:  pd.DataFrame) -> pd.Series:
    predefined_inputs = get_first_failed_tests(solutions_df=solutions_df, tests_df=tests_df).to_list()
    
    error_messages = []

    # Loop through each predefined input
    for input_value in predefined_inputs:
        # Create a generator for the predefined input
        inputs = iter([input_value])
        
        # Mock the input function to return the current predefined input
        def mock_input(prompt=""):
            return next(inputs)
        
        # Combine the student's code with the test case
        code_with_test = f"""
try:
    {student_code}
except Exception as e:
    raise e
"""
        try:
            # Replace the input function with mock_input
            global input
            original_input = input
            input = mock_input
            
            # Run the student's code
            exec(code_with_test)
            # If no exception, append None (no error)
            error_messages.append(None)
        
        except Exception as error:
            # If an error occurs, append the error message
            error_messages.append(str(error))
        
        finally:
            # Restore the original input function
            input = original_input
    
    # Return a Series where index is the predefined input and values are error messages (or None)
    return pd.Series(error_messages, index=predefined_inputs)



errors_series = get_error_messages_by_id(123, student_code, "./data/processed/test/solutions.xlsx", "./data/raw/test/tests.xlsx")
print(errors_series)

