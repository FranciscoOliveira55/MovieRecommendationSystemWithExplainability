import torch
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from skorch import NeuralNetRegressor
from sympy import false

from core import recommendation, prediction, data_loader, model, explainability, external_api, evaluation
from core.utils import write_log
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
from configs import ModelType, FilteringType, LEARNINGCURVEConfig
from sklearn.metrics import r2_score

if __name__ == '__main__':
    print("Recommendation System")
    print("This is a module for testing and debugging code")


    # Print max columns
    pd.set_option('display.max_columns', None)


    def cbf_prototype():
        # Gets the rated and unrated dataframes
        rated_df, unrated_df = data_loader.get_dataframes(
            ratings_user_id=1,
            clean_and_index_dfs=True,
            filtering_type=FilteringType.CONTENT_BASED_FILTERING
        )
        # print(f"Rated movies dataframe:\n {rated_df}\n")
        # print(f"Unrated movies dataframe:\n {unrated_df}\n")
        # Create the neural network model
        nn_model, _, _ = model.read_or_create_model(
            rated_movies_df=rated_df,
            target_column="rating",
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.CONTENT_BASED_FILTERING,
            user_id=1,
        )
        # Make rating predictions for unrated movies
        predicted_rating_df = prediction.predict_ratings_for_unrated_movies(
            unrated_movies_df=unrated_df,
            model=nn_model,
            model_type=ModelType.NEURAL_NETWORK
        )
        print(f"Predicted ratings dataframe:\n {predicted_rating_df}\n")
        # Recommend the top 5 unrated movies with the highest predicted rating
        x_number_of_movies_to_recommend = 5
        recommended_movies = recommendation.recommend_x_unrated_movies_with_the_highest_predicted_ratings(
            predicted_df=predicted_rating_df,
            x_number_of_movies_to_recommend=x_number_of_movies_to_recommend
        )
        print(
            f"Top {x_number_of_movies_to_recommend} recommended movies:\n {recommended_movies}\n")  # .name = the index value of the row
        # Now, lets add the explainable module
        # Generate SHAP explanation for the top recommended movie (local explainability)
        shap_values, expected_value = explainability.explain_predictions_with_shap_for_cbf(
            model=nn_model,
            rated_movies_dataset=rated_df,
            predicted_movies_to_explain=recommended_movies.iloc[[0]],  # Dataframe with only 1 row,
            model_type=ModelType.NEURAL_NETWORK
        )
        print(f"Expected model output (base value): {expected_value:.4f}")
        print("SHAP values for each feature:\n")
        feature_columns = rated_df.drop(columns=["rating"]).columns
        for feature_name, shap_val in zip(feature_columns, shap_values):
            print(f"{feature_name}: {shap_val:.4f}")

        # Filter SHAP values and feature names to exclude near-zero importance features
        filtered_shap_values, filtered_feature_names = explainability.filter_zero_shap_features(shap_values,
                                                                                                feature_columns.tolist())
        # Plot the SHAP values in a bar chart
        fig_bar_local = explainability.plot_shap_bar("Feature Importance (Local)(Bar Chart)", filtered_shap_values,
                                                     filtered_feature_names)
        # Plot the SHAP values in a pie chart
        fig_pie_local = explainability.plot_shap_pie("Feature Importance (Local)(Pie Chart)", filtered_shap_values,
                                                     filtered_feature_names)
        # Generate SHAP explanation for a batch o movies (global explainability)
        shap_values, expected_value = explainability.explain_predictions_with_shap_for_cbf(
            model=nn_model,
            rated_movies_dataset=rated_df,
            predicted_movies_to_explain=recommended_movies,
            model_type=ModelType.NEURAL_NETWORK
        )
        print(f"Expected model output (base value): {expected_value:.4f}")
        print("SHAP values for each feature:\n")
        feature_columns = rated_df.drop(columns=["rating"]).columns
        for feature_name, shap_val in zip(feature_columns, shap_values):
            print(f"{feature_name}: {shap_val:.4f}")
        # Filter SHAP values and feature names to exclude near-zero importance features
        filtered_shap_values, filtered_feature_names = explainability.filter_zero_shap_features(shap_values,
                                                                                                feature_columns.tolist())
        # Plot the SHAP values in a bar chart and show immediately
        fig_bar_global = explainability.plot_shap_bar("Feature Importance (Global)(Bar Chart)", filtered_shap_values,
                                                      filtered_feature_names)
        # Plot the SHAP values in a pie chart and show immediately
        fig_pie_global = explainability.plot_shap_pie("Feature Importance (Global)(Pie Chart)", filtered_shap_values,
                                                      filtered_feature_names)
        plt.show()


    def cf_prototype():

        selected_user_id = 1
        # Load rated and unrated data for a specific user
        rated_df: pd.DataFrame
        unrated_df: pd.DataFrame
        rated_df, unrated_df = data_loader.get_dataframes(
            ratings_user_id=selected_user_id,
            clean_and_index_dfs=False,
            filtering_type=FilteringType.COLLABORATIVE_FILTERING
        )

        print(f"rated_df.columns: {rated_df.columns}")
        print(f"rated_df.ndim: {rated_df.ndim}")
        print(f"rated_df.shape: {rated_df.shape}")
        print(f"unrated_df.columns: {unrated_df.columns}")
        print(f"unrated_df.ndim: {unrated_df.ndim}")
        print(f"unrated_df.shape: {unrated_df.shape}")
        print(f"rated_df: {rated_df}")
        print(f"unrated_df: {unrated_df}")

        # Prepare collaborative filtering model using embeddings
        nn_model: NeuralNetRegressor
        user_index: pd.Index
        item_index: pd.Index
        nn_model, user_index, item_index = model.read_or_create_model(
            rated_movies_df=rated_df,
            target_column="rating",
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.COLLABORATIVE_FILTERING,
            user_id=selected_user_id
        )

        # Prepare prediction df
        prediction_df: pd.DataFrame

        # Predict ratings for unrated movies
        prediction_df = prediction.predict_ratings_for_unrated_movies(
            unrated_movies_df=unrated_df,
            model=nn_model,
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.COLLABORATIVE_FILTERING,
            user_index=user_index,
            item_index=item_index
        )
        write_log(f"Predicted ratings dataframe:\n{prediction_df}\n")

        write_log("Making recommendations")
        # Recommend top N movies
        x_number_of_movies_to_recommend = 5
        recommended_movies = recommendation.recommend_x_unrated_movies_with_the_highest_predicted_ratings(
            predicted_df=prediction_df,
            x_number_of_movies_to_recommend=x_number_of_movies_to_recommend
        )
        write_log(f"Top {x_number_of_movies_to_recommend} recommended movies:\n{recommended_movies}\n")

        # Add tmdb IDs
        recommended_movies_with_tmdb_ids = data_loader.add_links_to_movies_df(recommended_movies)
        write_log(f"Recommended movies with tmdb ids:\n{recommended_movies_with_tmdb_ids}\n")

        # Add details to movies
        recommended_movies_with_tmdb_details = external_api.get_movies_details_from_tmdb(
            recommended_movies_with_tmdb_ids)
        write_log(f"Recommended movies with tmdb details:\n{recommended_movies_with_tmdb_details}\n")

        """
        local_cf_explanation, explanation_message=explainability.explain_local_predictions_for_cf(
            selected_user_id=selected_user_id,
            model=nn_model,
            rated_df_used_in_training=rated_df,
            user_index=user_index,
            predicted_movie_to_explain=recommended_movies.iloc[[0]]
        )
        print(f"local_cf_explanation: \n {local_cf_explanation}")
        """
        '''
        # Explanation
        """
        I Need:
        - the model
        - the recommended movie id (or df)
        - the rated_df used in training (need the embeddings)
        - the user indexs
        """

        nn_model: NeuralNetRegressor
        pytorch_model = nn_model.module_

        # 1. Filter users who rated the recommended movie
        recommended_movie_id = recommended_movies['movieId'].iloc[0]
        users_who_rated_movie = rated_df[rated_df['movieId'] == recommended_movie_id]['userId'].unique()

        # 2. Get embeddings only for those users
        all_user_vectors = pytorch_model.user_embedding.weight.detach().numpy()
        user_vectors_of_interest = all_user_vectors[
            [user_index.get_loc(u) for u in users_who_rated_movie if u in user_index]]

        # 3. Get the selected user's embedding vector
        selected_user_vector = pytorch_model.user_embedding(torch.tensor([selected_user_id])).detach().numpy()

        # 4. Compute cosine similarity only with these users
        similarities = cosine_similarity(selected_user_vector, user_vectors_of_interest)[0]

        # 5. Sort and get top_k similar users
        top_k = 5
        top_indices = similarities.argsort()[::-1][:top_k]

        # 6. Map back to the original user IDs
        similar_users_ids = users_who_rated_movie[top_indices]
        print(f"similar_users_ids: {similar_users_ids}")

        # 7. Get their similarity scores
        similarity_scores = similarities[top_indices]
        print(f"similarity_scores: {similarity_scores}")

        # 8. Get the ratings that these similar users gave to the recommended movie

        ratings_of_similar_users = rated_df[
            (rated_df['userId'].isin(similar_users_ids)) &
            (rated_df['movieId'] == recommended_movie_id)
            ][['userId', 'movieId', 'rating']]

        # Create a dictionary mapping userId -> similarity score
        similarity_dict = dict(zip(similar_users_ids, similarity_scores))
        # Add a new column 'similarity_score' by mapping userId through the dictionary
        ratings_of_similar_users['similarity_score'] = ratings_of_similar_users['userId'].map(similarity_dict)


        print(f"ratings_of_similar_users:\n {ratings_of_similar_users}")

        # Now you can show these users and similarity scores to explain the recommendation
        '''
        """
        # Vector of the selected user (shape: [1, embedding_dim])
        user_vector = pytorch_model.user_embedding(torch.tensor([selected_user_id])).detach().numpy()
        print(f"user_vector: {user_vector}")

        # All user vectors (shape: [n_users, embedding_dim])
        all_user_vectors = pytorch_model.user_embedding.weight.detach().numpy()
        print(f"all_user_vectors: {all_user_vectors}")

        # Compute cosine similarity between the selected user and all users
        similarities = cosine_similarity(user_vector, all_user_vectors)[0]

        # Get top N most similar users (excluding the selected user itself)
        top_k = 5
        similar_users_idx = similarities.argsort()[::-1][0:top_k + 1]  # 1 to exclude selected user
        print(f"similar_users_idx: {similar_users_idx}")

        # Similarity scores [1,0], 1 means 100% similar, 0 means, not similar at all
        users_similarity_scores = similarities[similar_users_idx]
        print(f"users_similarity_scores: {users_similarity_scores}")

        # Convert embedding indices back to original userIds
        similar_user_ids = user_index[similar_users_idx]

        # Filter the ratings dataframe to include only ratings from similar users for the specific movie
        ratings_of_similars = rated_df[
            (rated_df['userId'].isin(similar_user_ids)) &
            (rated_df['movieId'] == recommended_movies['movieId'].iloc[0])
            ]
        print(f"ratings_of_similar: {ratings_of_similars}")
        """
        """
        # CV Evaluation
        # Get input and output features
        x_input_features, y_output_feature, _, _, _, _ = model._prepare_training_data(
            rated_movies_df=rated_df,
            target_column="rating",
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.COLLABORATIVE_FILTERING
        )
        # Calculate cv
        cv_results = evaluation.cross_validate_model(
            model=nn_model,
            x_input_features=x_input_features,
            y_output_feature=y_output_feature,
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.COLLABORATIVE_FILTERING,
            selected_user_id=str(selected_user_id)
        )
        learning_curve = evaluation.calc_learning_curve_of_model(
            model=nn_model,
            x_input_features=x_input_features,
            y_output_feature=y_output_feature,
            learning_curve_scoring_metric_key=list(LEARNINGCURVEConfig.SCORING_METRICS.keys())[0],
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.COLLABORATIVE_FILTERING,
            selected_user_id=str(selected_user_id)
        )
        # Prepare learning curve for display
        print(cv_results)
        print(learning_curve)
        """


    def data_testing():
        selected_user = 118205.0
        selected_filtering_type = FilteringType.CONTENT_BASED_FILTERING

        uncleaned_rated_df1, uncleaned_unrated_df1 = data_loader.get_dataframes(
            ratings_user_id=-1,
            clean_and_index_dfs=False,
            filtering_type=selected_filtering_type
        )
        uncleaned_rated_df1 = uncleaned_rated_df1[uncleaned_rated_df1["userId"] == selected_user]

        # Filter dfs if its CBF (if its CF, then model.prepare_training_data takes care of it)
        rated_df1, _ = data_loader._clean_and_index_dfs(
            rated_df=uncleaned_rated_df1,
            unrated_df=uncleaned_unrated_df1,
            filtering_type=selected_filtering_type
        )
        x_input_features1, y_output_feature1, _, _, _, _ = model._prepare_training_data(rated_df1, "rating",
                                                                                        ModelType.NEURAL_NETWORK,
                                                                                        selected_filtering_type)
        x_train1, x_val1, y_train1, y_val1 = train_test_split(x_input_features1, y_output_feature1, test_size=0.2)
        model_base1 = LinearRegression()
        model_base1.fit(x_train1, y_train1)
        preds_base1 = model_base1.predict(x_val1)
        write_log(f"Baseline R2:, {r2_score(y_val1, preds_base1)}")
        write_log(f"rated_df filtered after size: {rated_df1.shape}")

        rated_df2, uncleaned_unrated_df2 = data_loader.get_dataframes(
            ratings_user_id=int(float(selected_user)),
            clean_and_index_dfs=False,
            filtering_type=selected_filtering_type
        )
        rated_df2, _ = data_loader._clean_and_index_dfs(
            rated_df=rated_df2,
            unrated_df=uncleaned_unrated_df2,
            filtering_type=selected_filtering_type
        )
        x_input_features2, y_output_feature2, _, _, _, _ = model._prepare_training_data(rated_df2, "rating",
                                                                                        ModelType.NEURAL_NETWORK,
                                                                                        selected_filtering_type)
        x_train2, x_val2, y_train2, y_val2 = train_test_split(x_input_features2, y_output_feature2, test_size=0.2)
        model_base2 = LinearRegression()
        model_base2.fit(x_train2, y_train2)
        preds_base2 = model_base2.predict(x_val2)
        write_log(f"Baseline R2:, {r2_score(y_val2, preds_base2)}")
        write_log(f"rated_df filtered before size: {rated_df2.shape}")

        rated_df1 = rated_df1.reset_index()
        rated_df2 = rated_df2.reset_index()
        movies_in_1_not_in_2 = rated_df1['movieId'][~rated_df1['movieId'].isin(rated_df2['movieId'])].unique()
        write_log(f"Movies in rated_df1 but not in rated_df2: {list(movies_in_1_not_in_2)}")
        movies_in_2_not_in_1 = rated_df2['movieId'][~rated_df2['movieId'].isin(rated_df1['movieId'])].unique()
        write_log(f"Movies in rated_df2 but not in rated_df1: {list(movies_in_2_not_in_1)}")
        write_log(f"rated1 columns {rated_df1.columns}")
        write_log(f"rated2 columns {rated_df2.columns}")
        genre_cols = [col for col in rated_df1.columns if col not in ['movieId', 'rating']]
        write_log(f"rated_df1 genre sum: {rated_df1[genre_cols].sum().to_dict()}")
        write_log(f"rated_df2 genre sum: {rated_df2[genre_cols].sum().to_dict()}")


    def hybrid_prototype():
        selected_user: int = 1

        # Prepare hybrid dataframes (cbf with cf data)
        hybrid_rated_df, hybrid_unrated_df = data_loader._get_dataframes_for_hybrid(
            selected_user_id=selected_user,
            clean_and_index_dfs=True
        )
        print(f"hybrid_rated_df: {hybrid_rated_df.shape}")
        print(f"hybrid_unrated_df: {hybrid_unrated_df.shape}")

        nn_model, _, _ = model.read_or_create_model(
            rated_movies_df=hybrid_rated_df,
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.HYBRID_FILTERING,
            user_id=selected_user,
        )

        predicted_rating_df = prediction.predict_ratings_for_unrated_movies(
            unrated_movies_df=hybrid_unrated_df,
            model=nn_model,
            model_type=ModelType.NEURAL_NETWORK
        )
        print(f"Predicted ratings dataframe:\n {predicted_rating_df.shape}\n")
        print(f"Predicted ratings dataframe:\n {predicted_rating_df.iloc[0]}\n")

        # Recommend the top 5 unrated movies with the highest predicted rating
        x_number_of_movies_to_recommend = 5
        recommended_movies = recommendation.recommend_x_unrated_movies_with_the_highest_predicted_ratings(
            predicted_df=predicted_rating_df,
            x_number_of_movies_to_recommend=x_number_of_movies_to_recommend
        )

        print(f"recommended_movies dataframe:\n {recommended_movies.shape}\n")
        print(f"recommended_movies dataframe:\n {recommended_movies.iloc[0]}\n")


        # Get input and output features
        x_input_features, y_output_feature, _, _, _, _ = model._prepare_training_data(
            rated_movies_df=hybrid_rated_df,
            target_column="rating",
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.HYBRID_FILTERING
        )

        cv_results:dict = evaluation.cross_validate_model(
            model=nn_model,
            x_input_features=x_input_features,
            y_output_feature=y_output_feature,
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.HYBRID_FILTERING,
            selected_user_id=str(selected_user)
        )

        learning_curve:dict = evaluation.calc_learning_curve_of_model(
            model=nn_model,
            x_input_features=x_input_features,
            y_output_feature=y_output_feature,
            learning_curve_scoring_metric_key=LEARNINGCURVEConfig.SCORING_METRICS['r2'],
            model_type=ModelType.NEURAL_NETWORK,
            filtering_type=FilteringType.HYBRID_FILTERING,
            selected_user_id=str(selected_user)
        )

        #print(f"cv_results:\n {cv_results}\n")
        #print(f"learning_curve:\n {learning_curve}\n")

        hybrid_uncleaned_rated_df, _ = data_loader._get_dataframes_for_hybrid(
            selected_user_id=selected_user,
            clean_and_index_dfs=False
        )
        #print(hybrid_uncleaned_rated_df)

        ratings_of_similar_users_df_sorted, explanation_text = explainability.explain_local_predictions_for_hyb(
            selected_user_id=selected_user,
            predicted_movie_to_explain=recommended_movies.iloc[[1]].reset_index(),
            #model=nn_model,
            rated_uncleaned_df=hybrid_uncleaned_rated_df,
        )
        print(f"ratings_of_similar_users_df_sorted:\n {ratings_of_similar_users_df_sorted}\n")
        print(f"explanation_text:\n {explanation_text}\n")


        # Generate SHAP explanation for a batch o movies (global explainability)
        shap_values, expected_value = explainability.explain_predictions_with_shap_for_cbf(
            model=nn_model,
            rated_movies_dataset=hybrid_rated_df,
            predicted_movies_to_explain=recommended_movies,
            model_type=ModelType.NEURAL_NETWORK
        )
        print(f"Expected model output (base value): {expected_value:.4f}")
        #print(f"SHAP values:\n {shap_values}")
        feature_columns = hybrid_rated_df.drop(columns=["rating"]).columns

        for feature_name, shap_val in zip(feature_columns, shap_values):
            print(f"{feature_name}: {shap_val}")


        # Filter SHAP values and feature names to exclude near-zero importance features
        filtered_shap_values, filtered_feature_names = explainability.filter_zero_shap_features(shap_values,
                                                                                         feature_columns.tolist())
        import matplotlib
        matplotlib.use("TkAgg")  # ou "Qt5Agg" se tiveres Qt

        # Plot the SHAP values in a bar chart and show immediately
        fig_bar_global = explainability.plot_shap_bar("Feature Importance (Global)(Bar Chart)", filtered_shap_values,
                                                      filtered_feature_names)
        # Plot the SHAP values in a pie chart and show immediately
        fig_pie_global = explainability.plot_shap_pie("Feature Importance (Global)(Pie Chart)", filtered_shap_values,
                                                      filtered_feature_names)
        plt.show()


    def data_testing2():
        selected_user: int = 118205

        # Prepare hybrid dataframes (cbf with cf data)
        hybrid_rated_df, hybrid_unrated_df = data_loader._get_dataframes_for_hybrid(
            selected_user_id=selected_user,
            clean_and_index_dfs=False
        )
        print(hybrid_rated_df.shape)
        print(hybrid_rated_df['userId'].astype(int).unique())
        print(hybrid_rated_df['userId'].astype(int).nunique())
        print(hybrid_unrated_df.shape)
        #print(hybrid_rated_df[hybrid_rated_df["movieId"] == 118768])
        #print(hybrid_unrated_df[hybrid_unrated_df["movieId"] == 118768])



    data_testing2()
