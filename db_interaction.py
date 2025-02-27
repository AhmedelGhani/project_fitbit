

# ----------------------------------------------------------------------------
# PART 3: Interacting with the database
# ----------------------------------------------------------------------------

def classify_users(daily_df):
   

    usage_counting = daily_df.groupby('Id').size().reset_index(name='UsageCount')

    def classify(usage):
        if usage <= 10:
            return "Light"
        elif 11 <= usage <= 15:
            return "Moderate"
        else:
            return "Heavy"

    usage_counting['Class'] = usage_counting['UsageCount'].apply(classify)
    return usage_counting[['Id', 'Class']]

