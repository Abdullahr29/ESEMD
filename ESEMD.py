import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
from termcolor import colored
from sklearn.decomposition import PCA
import pickle
import argparse
import os
import time
import traceback

np.random.seed(1)

class SparseEigenmotionDecomposition: 

    def __init__(self, args):

        self.learning_iterations = 40000 #modify as needed
        ### ====== Hyperparameters ====== ###
        # Determines the number of timesteps from which we extract local PCs
        self.ws = args.ws
        self.prqt = args.prqt # Reconstruction quality threshold required for initial PCs
        self.srqt = args.srqt
        self.st = args.st
        self.improvement_counter = 0
        self.failed_initial_pc = True

        self.dim_flag = args.dimension_flag

        ### Visualization/Plotting 
        self.visualise = False # NOTE: Set this flag false if you don't want to see plots

        ### Various flags and containers to make the program run, (you can ignore these)
        self.dictionary = {}
        self.reconstruction_score_dictionary = {}
        self.no_new_PC_found = False
        self.first_eigenmotion_found = False
        self.below = True


    def plot_eigenmotion(self, eigenmotion):
        eig_sum = np.sum(eigenmotion,axis=0)
        for k in range(len(eigenmotion)):
            plt.subplot(2,3,k+1)
            plt.title("Component " + str(k+1))
            plt.bar(np.arange(len(eigenmotion[k])), eigenmotion[k]) 
        plt.subplot(2,3,6)
        plt.bar(np.arange(len(eig_sum)), eig_sum)
        plt.title("Summed eigenmotions")

    def extract_eigenmotions(self, data):
        """

        """

        try:
            self.reconstruction_data = data # Always use the whole dataset for reconstruction
            self.ROI_data = data # Have another identical dataset from where we remove bad ROIs for future ROI selections
            self.global_reconstruction_scores = np.zeros(len(self.reconstruction_data))

            mean = np.mean(data,axis=0) # Get the mean 
            self.mean = mean




            iteration_start = 0
            iteration_end = self.learning_iterations


            print("=====================================================================")
            print(colored("\n The program is starting with the following parameters:",'green'))
            print("\n - Window size: {}".format(self.ws))
            print("\nDataset shape: {}".format(data.shape))


            time.sleep(1)

            for i in range(iteration_start,iteration_end):

                # This mask will be used to ignore indices where ROI failed to create a satisfying eigenmotion
                self.ignore_mask = np.zeros(data.shape[0])

                plt.close('all')

                print("\nLearning iteration: ", i, "\nAverage reconstruction score: ", 
                    np.mean(self.global_reconstruction_scores[self.global_reconstruction_scores !=2] ), "\nNumber of eigenmotions in the dictionary", 
                    len(self.dictionary))

                self.learning_iter = i
                self.improvement_counter += 1

                # Step 1: Find ROI
                if i == 0: # First iteration we pick random ROI, pass in 0 idx as placeholder, it will be replaced by random idx
                    self.random_roi = True
                    self.idx = 0
                roi = self.select_roi(self.ws)

                # Step 2: Find the n most important PCs 
                initial_pcs = self.find_initial_pcs(roi)

                # print("initial_pcs: ", initial_pcs.n_components , end="\r")

                # Step 3: Check signifigance of each PC and add the PCs that pass the significance threshold temporarily to a dictionary
                if self.failed_initial_pc == False:
                    self.check_pc_significance(initial_pcs)
                # If we failed to find first PC we pick a random ROI again
                else:
                    continue
 
                # If improvement hasnt occurred 1000 iterations and learning iterations is over 5k we stop
                if self.improvement_counter > 3000 and self.learning_iter > 5000:

                    print("Extraction finished")
                    self.reduce_interference()
                    return self.dictionary

            self.reduce_interference()
            return self.dictionary
        except:
            print("Error in the main loop")
            print(traceback.format_exc())



    def select_roi(self, window_size,idx=None):
        """
        - Selects initial ROI randomly
        - Selects subsequent ROI's based on the lowest reconstruction score

        Returns
        -------
        ndarray data(samples, dimensions)
            Returns a reduced dataset of actions, ROI from which the PCs were extracted is removed from the data

        ndarray roi(samples, dimensions)
            Returns a ROI from which the next eigenmotions will be extracted
        """
        try:

            # Deal with an edge case where idx is at the end of the data
            if self.idx > len(self.ROI_data) - window_size:
                self.idx = len(self.ROI_data) - window_size


            # If we are on first iteration we select the ROI randomly
            if self.random_roi == True:
                idx = np.random.choice(range(self.ROI_data.shape[0]-window_size)) # Find a random index
                #idx = 0
                roi = self.ROI_data[idx:idx+window_size,:] # Select the region of interest with the random index
            # For subsequent ROI we select around the time-point with lowest reconstruction score
            else: 
                roi = self.ROI_data[self.idx:self.idx+window_size,:] 

            if self.visualise == True:
                plt.title("Selected ROI")
                plt.ylabel("Velocity")
                plt.xlabel("Timestep")
                plt.plot(np.arange(len(roi)),roi)
                plt.show()

            return roi
        except:
            print("Error in picking the next ROI")
            print(traceback.format_exc())


    # This function finds the eigenmotions/pcs from the ROI
    def find_initial_pcs(self, roi, threshold = 10e-4, show=True):
        """
        """
        try:

            for i in range(len(roi)):

                ### Step 1: Find the n most significant PCs 
                pca = PCA(n_components=i+1)
                pca.fit(roi)
                latent = np.dot(roi, pca.components_.T)
                recon = np.dot(latent,pca.components_) 

                lGOF = 1 - np.var(roi-recon)/np.var(roi)


                # If we have found PCs that satisfy the reconstruction quality, we return them
                if lGOF > self.prqt:
                    self.failed_initial_pc = False
                    break

                # if self.dim_flag:
                #     break

                # If we fail to find satisfying PCs, we pick the next lowest region
                if i == len(roi) - 1:
                    # Temporarily set the score at index which fails to 2, which will not be considered in the future sampling
                    self.global_reconstruction_scores[self.idx] = 2

                    self.idx = np.argmin(self.global_reconstruction_scores)
                    #self.idx = self.find_lowest_window()
                    break

            return pca
        except:
            print("Error in initial PC extraction")
            print(traceback.format_exc())
    
    def get_percentage_successful_datapoints(self, reconstruction_scores, flag):

        # Get indices of datapoints that achieved reconstruction threhsold
        if flag == True: # if primary threshold
            success_indices = np.where(reconstruction_scores > self.prqt)[0]
        else: # if secondary threshold
            success_indices = np.where(reconstruction_scores > self.srqt)[0]

        # Get the number of successful datapoints
        highest_success_count = len(success_indices)

        # Compute the percentage significance of the PCs 
        success_significance = highest_success_count / len(self.reconstruction_data)

        return success_indices, success_significance


    def count_sequential_successes(self, success_array):
        sequential_counter = 0
        for i in range(len(success_array)):
            if i+1 == len(success_array):
                break
            if success_array[i+1] - success_array[i] == 1:
                sequential_counter+=1
        return sequential_counter

    # Check PC significance across the whole data
    def check_pc_significance(self,pcs):
        """
        1. Reconstruct the whole data
        2. Find timesteps where reconstruction was above threshold
        3. Check if the number of relevant datapoints for this PC is above a threshold
            - If yes, continue
            - If no, update the reconstruction scores array such that the current index is not picked again
        4. Pool the relevant datapoints and extract an eigenmotion
        5. Find the indices and the percentage of reconstructions above threshold with that new eigenmotion
        6. If reconstructions above eigenmotion reconstruction threshold, find the percentage of timesteps this eigenmotion is the best 
            at recontsruction compared to other eigenmotions.
        6. 
        """
        try: 

            # Reconstruct 
            latent_pc = np.dot(self.reconstruction_data, pcs.components_.T)
            recon = np.dot(latent_pc,pcs.components_)
            # Compute reconstruction scores
            gGOF = 1 - np.var(self.reconstruction_data-recon, axis=1)/np.var(self.reconstruction_data,axis=1)

            # Get the percentage of datapoints that hit the Prqt
            prqt = False # set this flag true, we look for reconstructions above primary threshold
            successes, success_percentage = self.get_percentage_successful_datapoints(gGOF, prqt)

            # Average reconstruction score
            avg_rec = np.mean(gGOF)
            avg_rec = np.round(avg_rec, 3)
                
            # Check whether enough datapoints passed a threshold for the PC to be relevant
            p_success = len(successes) / len(self.reconstruction_data)
            relevance_threshold = self.st * 0.7 #why is this happening

            # If there are too few primary successes we don't waste time on the next steps
            if p_success < relevance_threshold:
                # This index failed to produce good enough PC, make sure we dont pick this index again
                self.global_reconstruction_scores[self.idx] = 2
                self.idx = np.argmin(self.global_reconstruction_scores)
                #self.idx = self.find_lowest_window()
                return

            # Extract new eigenmotion
            pooled_datapoints = self.reconstruction_data[successes]
            if(not self.dim_flag):
                pca = PCA(n_components = pcs.n_components) # Construct PCA, why do we assume/extract this pc with the same dimensionality
            else:
                pca = PCA(n_components = 1)
            pca.fit(pooled_datapoints)

            # Reconstruct with new eigenmotion
            latent_eig = np.dot(self.reconstruction_data, pca.components_.T)
            eig_recon = np.dot(latent_eig,pca.components_)

            # Compute reconstruction scores
            eig_gGOF = 1 - np.var(self.reconstruction_data-eig_recon, axis=1)/np.var(self.reconstruction_data,axis=1)
            avg_eig_gGOF = np.mean(eig_gGOF)
            avg_eig_gGOF = np.round(avg_eig_gGOF, 3)

            prqt = False
            successes, success_percentage = self.get_percentage_successful_datapoints(eig_gGOF, prqt)


            # If secondary reconstruction quality significance is above threshold, then we proceed to check whether it's also significant when compared to other eigenmotions
            if success_percentage > self.st:
                # Find indices where the new eigenmotion is better than the old eigenmotions
                better_indices = np.where(eig_gGOF[successes] > 
                    self.global_reconstruction_scores[successes] ) 
                # Compute the percentage of indices where the new eigenmotion is best
                significance = len(better_indices[0]) / len(self.reconstruction_data)
                # If the new eigenmotion is the best in significant number of timesteps, then we add it as a new eigenmotion
                if significance > self.st:
                    print(colored("Good eigenmotion found", "green" ))
                    # Add eigenmotion to dictionary
                    self.dictionary["eigenmotion_"+str(self.learning_iter)] = pca.components_
                    # Add eigenmotion reconstructions to dictionary to keep track of how each eigenmotion contributes to global score
                    self.reconstruction_score_dictionary["eigenmotion_"+str(self.learning_iter)] = eig_gGOF
                    self.improvement_counter = 0 # Reset improvement counter

                    # We have found fist eigenmotion and have reconstruction scores, therefore we no longer have to select ROI randomly
                    self.random_roi = False

                    # Update global reconstruction scores
                    self.global_reconstruction_scores = np.where(eig_gGOF > self.global_reconstruction_scores, 
                        eig_gGOF, self.global_reconstruction_scores)

                    # Find the lowest reconstruction idx
                    self.idx = np.argmin(self.global_reconstruction_scores)
                    #self.idx = self.find_lowest_window()

                    # Remove eigenmotions that have potentially lost significance as a result of the new addition
                    for i in self.dictionary.copy():
                        # Reconstruct 
                        latent_pc = np.dot(self.reconstruction_data, self.dictionary[i].T)
                        recon = np.dot(latent_pc, self.dictionary[i])
                        # Compute reconstruction scores
                        reconstruction_scores = 1 - np.var(self.reconstruction_data-recon, axis=1)/np.var(self.reconstruction_data,axis=1)

                        # Find indices where eigenmotion is the best
                        best_indices = np.where(reconstruction_scores >= self.global_reconstruction_scores)
                        eig_significance = len(best_indices[0]) / len(self.reconstruction_data)
                        
                        #eig_similarity = np.linalg.norm(eig_recon - recon)

                        # If significance is less than threshold then remove the eigenmotion from dictionary
                        #  or (eig_similarity < 5 and eig_similarity > 1)
                        if eig_significance < self.st: 
                            del self.dictionary[i]
                            del self.reconstruction_score_dictionary[i]
                            print(colored("Eigenmotion deleted from dictionary", "red"))

                            # Update global reconstruction scores
                            for rs in range(len(self.global_reconstruction_scores)):
                                # Find the highest score for each global score index with remaining eigenmotions
                                highest = 0
                                for eig in self.reconstruction_score_dictionary:
                                    score = self.reconstruction_score_dictionary[eig][rs]
                                    if score > highest:
                                        highest = score

                                self.global_reconstruction_scores[rs] = highest 
                            # Find the lowest reconstruction idx
                            self.idx = np.argmin(self.global_reconstruction_scores)
                            #self.idx = self.find_lowest_window()

                else:
                    self.global_reconstruction_scores[self.idx] = 2
                    self.idx = np.argmin(self.global_reconstruction_scores)
                    #self.idx = self.find_lowest_window()
                    # print(colored("Good Eigenmotion not found", "red"), end="\r")
                    pass
            else:
                self.global_reconstruction_scores[self.idx] = 2
                self.idx = np.argmin(self.global_reconstruction_scores)
                #self.idx = self.find_lowest_window()
                # print(colored("Good Eigenmotion not found", "red"), end="\r")
                pass



            return pca.components_
        except:
            print("Error in PC significance estimation")
            print(traceback.format_exc())

    def test_eigenmotions_full(test_data, dictionary):
        #function to throughly test each atom of dictionary
        reconstruction_score_matrix = np.zeros((len(dictionary), len(test_data)))
        reconstructions = np.zeros((len(dictionary), len(test_data), len(data_vel[0,:])))
        best_reconstruction = np.zeros((len(test_data), len(data_vel[0,:])))

        for i in range(len(test_data)):
            idx = 0
            for j in dictionary:
                #(1, 69) dot (69, 8)
                latent = np.dot(test_data[i,:], dictionary[j].T)
                #(1, 8) dot (8, 69)
                recon = np.dot(latent, dictionary[j])
                reconstructions[idx, i, :] = recon
                reconstruction_scores = 1 - np.var(test_data[i,:]-recon)/np.var(test_data[i,:])
                reconstruction_score_matrix[idx,i] = reconstruction_scores
                idx+=1
    
        greedy_eigenmotion = np.argmax(reconstruction_score_matrix,axis=0)

        idx = 0
        for i in greedy_eigenmotion:
            best_reconstruction[idx, :] = reconstructions[i, idx, :]
            idx+=1

        best_scores = np.max(reconstruction_score_matrix,axis=0)

        #returns: matrix with reconstruction score at each timestep with each atom, each time point reconstructed with its best atom, 
        # vector of best score per time point, vector of best eigenmotion per time point 
        return reconstruction_score_matrix, best_reconstruction, best_scores, greedy_eigenmotion

    def reduce_interference(self):

        print("\nBeginning interference reduction step")

        recon_matrix, best_recon, best_score, eigenmotion_idx = test_eigenmotions_full(self.reconstruction_data, self.dictionary)

        print('Pre interference reduction reconstruction Score: ' + str(np.mean(best_score)))

        #extract indices which previously could not be reconstructed
        two_idx = np.where(self.global_reconstruction_scores == 2)[0]
        #extract indices with poor reconstruction, 0.1 can be parametrised
        low_idx = np.where(best_score < np.mean(best_score)-0.1)[0]
        #extract indices where reconstruction with different best and second best eigenmotion was very similar
        recon_sorted = np.sort(recon_matrix, axis=0)
        recon_sorted = np.flip(recon_sorted, axis=0)
        sim_idx = np.where(recon_sorted[0,:] - recon_sorted[1,:] < 0.05)[0]

        #intersection of low and similar idx
        reduce_idx = np.intersect1d(low_idx, sim_idx)
        #union of previous and non-reconstructed idx
        reduce_idx = np.union1d(reduce_idx, two_idx)

        print("\nAttempting to reduce interference at "+ str(len(reduce_idx)) + " time points")

        time.sleep(1)

        #reset previously mentioned idx
        self.global_reconstruction_scores[reduce_idx] = 0
        #self.windowed_reconstruction_scores[reduce_idx] = 0 

        #decrease thresholds, values can be parametrised
        self.prqt = self.prqt - 0.1
        self.srqt = self.srqt - 0.1
        self.st = self.st/2

        for i in range(0,10000):


            print("\nLearning iteration: ", i, "\nAverage reconstruction score: ", 
                np.mean(self.global_reconstruction_scores[self.global_reconstruction_scores !=2] ), "\nNumber of eigenmotions in the dictionary", 
                len(self.dictionary))

            self.learning_iter = i
            self.improvement_counter += 1

            # Step 1: Find ROI
            if i == 0: # First iteration we pick random ROI, pass in 0 idx as placeholder, it will be replaced by random idx
                self.random_roi = True
                self.idx = 0
            roi = self.select_roi(self.ws)

            # Step 2: Find the n most important PCs 
            initial_pcs = self.find_initial_pcs(roi)

            # print("initial_pcs: ", initial_pcs.n_components , end="\r")

            # Step 3: Check signifigance of each PC and add the PCs that pass the significance threshold temporarily to a dictionary
            self.check_pc_significance(initial_pcs)

 
            # If improvement hasnt occurred 1000 iterations and learning iterations is over 5k we stop
            if self.improvement_counter > 5000 and self.learning_iter > 5000:

                print("Extraction finished")
                return self.dictionary

        return

    def find_lowest_window(self):
        #function for windowed lowest ROI search, however is too slow and needs further work

        # if((self.idx < self.global_reconstruction_scores.size-3) and self.below):
        #     return self.idx + 1      
        # else:
        #     self.below = False
        
        self.windowed_reconstruction_scores = self.global_reconstruction_scores

        two_idx = np.where(self.global_reconstruction_scores == 2)[0]
        self.windowed_reconstruction_scores[two_idx] = 0
        #print("cp2")
        for i in range(self.windowed_reconstruction_scores.size-self.ws+1):
            self.windowed_reconstruction_scores[i] = np.sum(self.windowed_reconstruction_scores[i:(i+self.ws)])/self.ws
        #print("cp3")
        for i in range(self.windowed_reconstruction_scores.size-self.ws+1, self.windowed_reconstruction_scores.size-1):
            self.windowed_reconstruction_scores[i] = np.sum(self.windowed_reconstruction_scores[i:-1])/len(self.windowed_reconstruction_scores[i:-1])
        #print("cp4")
        return np.argmin(self.windowed_reconstruction_scores)
        

def calculate_reconstruction_score(data,reconstruction):
    score = 1 - np.var(data-reconstruction)/np.var(data)
    return score 

def calculate_reconstruction_error(data, reconstruction):
    error = np.mean(np.sum((reconstruction - data) ** 2) / np.sum(data ** 2))
    return error


def test_eigenmotions_full(test_data, dictionary):
    #function to throughly test each atom of dictionary
    reconstruction_score_matrix = np.zeros((len(dictionary), len(test_data)))
    reconstructions = np.zeros((len(dictionary), len(test_data), len(data_vel[0,:])))
    best_reconstruction = np.zeros((len(test_data), len(data_vel[0,:])))

    for i in range(len(test_data)):
        idx = 0
        for j in dictionary:
            #(1, 69) dot (69, 8)
            latent = np.dot(test_data[i,:], dictionary[j].T)
            #(1, 8) dot (8, 69)
            recon = np.dot(latent, dictionary[j])
            reconstructions[idx, i, :] = recon
            reconstruction_scores = 1 - np.var(test_data[i,:]-recon)/np.var(test_data[i,:])
            reconstruction_score_matrix[idx,i] = reconstruction_scores
            idx+=1
    
    greedy_eigenmotion = np.argmax(reconstruction_score_matrix,axis=0)

    idx = 0
    for i in greedy_eigenmotion:
        best_reconstruction[idx, :] = reconstructions[i, idx, :]
        idx+=1


    best_scores = np.max(reconstruction_score_matrix,axis=0)
    
    #returns: matrix with reconstruction score at each timestep with each atom, each time point reconstructed with its best atom, 
    # vector of best score per time point, vector of best eigenmotion per time point 
    return reconstruction_score_matrix, best_reconstruction, best_scores, greedy_eigenmotion


if __name__ == "__main__":

    # Collect input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--ws", type=int, default=8)
    parser.add_argument("--prqt", type=float, default=0.75)
    parser.add_argument("--srqt", type=float, default=0.65)
    parser.add_argument("--st", type=float, default=0.01)
    parser.add_argument("--data_size", type=int, default = 7000)
    parser.add_argument("--dimension_flag", type=bool, default=True)
    args = parser.parse_args()


    print("Reading in the data...")


    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    #rel_path = 'input_data/Novice_ang_Data.csv' #9194
    #path to input file
    rel_path = 'input_data/AngVel_Data.csv' #7362
    abs_file_path = os.path.join(script_dir, rel_path)

    #path to output pickle for saving dictionary
    abs_output_path = 'output_data/eigenaction_angvel1dim_75_65_1test.pickle'
    abs_output_path = os.path.join(script_dir, abs_output_path)

    actions = np.genfromtxt(abs_file_path, delimiter=',')
    actions_check = np.array(actions)
    print("This is the shape of the input data: "+ str(actions_check.shape))
    data_vel = actions
    mean = np.mean(data_vel,axis=0)
    data_vel = data_vel - mean

    args.data_size = len(data_vel) - 1
    print(args.data_size)

    #grid search params
    # win_sizes = [6, 8, 10, 14, 20, 24]
    # prqt = [0.75, 0.8, 0.85, 0.9, 0.95]
    # st= [0.01,...]
    # srqt = [-0.2, -0.15, -0.1, -0.05, 0]
    # grid_search = {'Win Size' : [], 'PRQT' : [], 'SRQT' : [], 'ST' : [], 'Data_Size' : [], 'Dict' : [], 'OMP R' : [], 'OMP E' : []} #structure is: parameters, dictionary size, OMP R, OMP Error, LL R, LL Error
    # gs_results = pd.DataFrame(grid_search)


    #train_test_split on training data instead of constant
    training_data = data_vel[0:args.data_size] # select a subset of data
    curr = time.time()
    SED = SparseEigenmotionDecomposition(args)
    eigenmotions_dict = SED.extract_eigenmotions(training_data)

    recon_matrix, best_recon, best_score, eigenmotion_idx = test_eigenmotions_full(data_vel, eigenmotions_dict)

    with open(abs_output_path, "wb") as output_file:
        pickle.dump(eigenmotions_dict, output_file)

    print('Time taken: ', time.time()-curr) 

    print('Eigenmotion Dimensions: ')
    for i in eigenmotions_dict:
        print(eigenmotions_dict[i].shape)

    plt.imshow(recon_matrix, aspect='auto')
    plt.colorbar()
    plt.show()
    print('Reconstruction Score: ' + str(np.mean(best_score)))
    error = calculate_reconstruction_error(data_vel, best_recon)
    print('Reconstruction Error: ' + str(error))
    plt.scatter(list(range(0, len(eigenmotion_idx))), eigenmotion_idx)
    plt.show()
    plt.plot(np.bincount(eigenmotion_idx))
    plt.show()

    print("Eigenmotion extraction finished")
 