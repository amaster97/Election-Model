import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import LinearRegression
import string
from scipy.interpolate import interp1d
import matplotlib.animation as animation

def make_voters():
    """Change me to make new voters"""
    x1 = np.random.normal(25, 3, 1200)
    x2 = np.random.normal(75, 3, 1200)
    x3 = np.random.normal(50, 10, 100)
    return np.concatenate([x1, x2,x3])
    
    
    
def make_init_candidates():
    """Change me to make new candidate initial positions"""

    candidates = []

    for i in range(100):
        candidates.append(np.random.uniform(100))
        
        
    return candidates


    
def interpolate(candidate_pos, votes_won, pos_max, candidate_names=None):
    
    if not candidate_names:
        candidate_names = np.arange(len(candidate_pos))
    
    results_df = pd.DataFrame({
        'Name': candidate_names,
        'Positions': candidate_pos,
        'Votes': votes_won
    })
    
    
    results_df = results_df.sort_values(by=['Positions'])
    
    results_df['Cum_Votes'] = results_df['Votes'].cumsum()
    
    
    half_thresh = results_df['Votes'].sum()/2
    
    if half_thresh < results_df['Cum_Votes'].min():
        opt_pos = results_df['Positions'].min() / 2
    
    else:
        interpol = interp1d(results_df['Cum_Votes'], results_df['Positions'])
        opt_pos = interpol(half_thresh)
    
    
    
    return results_df, opt_pos



class election_model_1:

    def __init__(self, voters, min_cand_dist, voter_diff_stop_ratio, benefit_func= None, allow_cand_flipping=False, pos_max=100, voting_thresh=None, max_iterations=20):
        
        self.VOTERS = voters
        
        if not benefit_func:
            self.BENEFIT_FUNC = (lambda c, v: 1/np.abs(v-c))
        else:
            self.BENEFIT_FUNC = benefit_func
        
        self.MIN_CANDIDATE_DIST = min_cand_dist
        self.VOTER_DIFF_STOP_RATIO = voter_diff_stop_ratio
        self.ALLOW_CANDIDATE_FLIP = allow_cand_flipping
        self.POS_MAX = pos_max
        
        self.VOTING_THRESH = voting_thresh
        self.MAX_ITERS = max_iterations
        self.clear_results_dict()


    def count_points(self, candidates):


        utils = np.zeros(shape=(len(candidates), len(self.VOTERS)))
        wins = np.zeros(len(candidates))


        for i in range(len(candidates)):
            utils[i] = self.BENEFIT_FUNC(candidates[i], self.VOTERS)

        for r in utils.T:

            if self.VOTING_THRESH:
                ## this person won't bother voting if their utilities from first or last choice candidate is less than threshold
                utils_diff = np.max(r) - np.min(r)
                if utils_diff > self.VOTING_THRESH:
                    wins[np.argmax(r)] += 1

            else:
                wins[np.argmax(r)] += 1

        return wins 
    

    def show_many_candidates(self, candidates):
        
        print('One Run Interpolation with Many Candidates')

        plt.hist(self.VOTERS, color='green')

        for c in candidates:
            plt.axvline(x=c, color='orange')
        plt.show()
        
        res = interpolate(candidates, self.count_points(candidates), self.POS_MAX)

        print('Optimal Political Positioning from Linear Interpolation:', res[1])
        return res


    def clear_results_dict(self):
        self.results_dict = {
            
            'A': [],
            'B': [],
            'Votes_A': [],
            'Votes_B': []
            
        }
   
    
    def fill_results_dict(self, A, B, v_A, v_B):
        
        self.results_dict['A'].append(float(A))
        self.results_dict['B'].append(float(B))
        self.results_dict['Votes_A'].append(float(v_A))
        self.results_dict['Votes_B'].append(float(v_B))

        
        
        
    def generate_candidate_movements(self, A, B, clear_position_arr=True):
        """
        Rules of the game:  Each canddiate, A,B starts off at some point and then compete for votes
        every round the loser takes on the position of the interpolated optimal position
        repeat until they are within epsilon of each other

        """
        
        ## reset positions
        
        if clear_position_arr:
            self.clear_results_dict()
        
        ## add positions
        result = self.count_points([A, B])
        interp_df, opt_pos = interpolate([A, B], result, self.POS_MAX, ['A', 'B'])
        self.fill_results_dict(A, B, result[0], result[1])

        trials = 0
        while trials < self.MAX_ITERS:
            
            if self.ALLOW_CANDIDATE_FLIP == False and abs(A-B) < self.MIN_CANDIDATE_DIST:
                break
            
            trials += 1
            
            loser = np.argmin(result)

            if np.sum(result) > 0 and (abs(result[0] - result[1]) / np.sum(result)) < self.VOTER_DIFF_STOP_RATIO:
                ## A and B both feel comepelled to move more center if it's close election
                
                midpt = (A+B)/2
                A = (A+midpt) / 2
                B = (B+midpt) / 2
                
            else:

                if loser == 0:
                    A = opt_pos
                    if self.ALLOW_CANDIDATE_FLIP == False and A > B:
                        A = B - self.MIN_CANDIDATE_DIST

                else:
                    B = opt_pos
                    if self.ALLOW_CANDIDATE_FLIP == False and A > B:
                        B = A + self.MIN_CANDIDATE_DIST
                    
                    
                    
            
                    
            ## update new result position arr
            result = self.count_points([A, B])
            interp_df, opt_pos = interpolate([A, B], result, self.POS_MAX, ['A', 'B'])
            self.fill_results_dict(A, B, result[0], result[1])
            
            print(A, B, result)
            
            ### if just oscillating
            if self.oscilating_candidate(self.results_dict['A']) and self.oscilating_candidate(self.results_dict['B']):
                break
                
            
            

        if self.ALLOW_CANDIDATE_FLIP == False:
            return result, A, B
    
        ## IF you fail to get parity, then the loser has to jump over to the winner's side
        else:
            loser = np.argmin(result)
            loser_votes = np.min(result)

            if loser == 0:
                # A lost
                if A < B:
                    A = (np.max(self.VOTERS) + B)/2
                else:
                    A = (np.min(self.VOTERS) + B)/2        

            else:
                # B lost
                if B < A:
                    B = (np.max(self.VOTERS) + A)/2
                else:
                    B = (np.min(self.VOTERS) + A)/2
            
            ## stop flipping infinite loop
            self.ALLOW_CANDIDATE_FLIP = False
            new_res, new_A, new_B = self.generate_candidate_movements(A ,B, False)
            new_loser_votes = new_res[loser]


            if new_loser_votes > loser_votes:
                return new_res, new_A, new_B

            else:
                
                self.fill_results_dict(A, B, result[0], result[1])
                return result, A, B        

        
            

        return result, A, B
    
    
    
    def oscilating_candidate(self, pos_vec):
        
        if len(pos_vec) <= 5:
            return False
        
        if abs(pos_vec[-1] -pos_vec[-3]) < self.MIN_CANDIDATE_DIST and abs(pos_vec[-1] - pos_vec[-5]) < self.MIN_CANDIDATE_DIST:
            return True
        
        
        return False

    def summarize_generate_candidate_movements(self, A, B):
        
        print('Two Candidate Convergence')
        
        plt.hist(self.VOTERS, color='green')
        plt.axvline(x=A, label='Initial A', linestyle='--', color='Blue')    
        plt.axvline(x=B, label='Initial B', linestyle='--', color='Red')

        final_votes, final_A, final_B = self.generate_candidate_movements(A, B)

        plt.axvline(x=final_A, label='Final A', linestyle='-', color='Blue')    
        plt.axvline(x=final_B, label='Final B', linestyle='-', color='Red')

        plt.legend()
        plt.show()
        
        final_interpolation = interpolate([final_A, final_B], final_votes, self.POS_MAX)[0]
        
        print(final_interpolation)
        
        return final_interpolation

    

    
    
class election_model_2(election_model_1):
    
    
    def __init__(self,max_iterations, voters, min_cand_dist, voter_diff_stop_ratio, benefit_func= None, allow_cand_flipping=False, pos_max=100, voting_thresh=None, loser_mvt=1):
        
        super().__init__(voters, min_cand_dist, voter_diff_stop_ratio, benefit_func, allow_cand_flipping, pos_max, voting_thresh)
        self.LOSER_MVT = loser_mvt
        

    

    
    def generate_candidate_movements(self, A, B):
        
        cur_A = A
        cur_B = B
        
        ## clear movement and append
        self.clear_results_dict()

        
        result = self.count_points([cur_A, cur_B])
        interp_df, opt_pos = interpolate([cur_A, cur_B], result, self.POS_MAX, ['A', 'B'])
        self.fill_results_dict(cur_A, cur_B, result[0], result[1])
        
        
        trials = 0
        
        prev_movement = None
        while trials < self.MAX_ITERS:
            if self.ALLOW_CANDIDATE_FLIP == False and abs(A-B) < self.MIN_CANDIDATE_DIST:
                break
            trials+=1
    
            
            if np.sum(result) > 0 and (abs(result[0] - result[1]) / np.sum(result)) < self.VOTER_DIFF_STOP_RATIO:
                ## A and B both feel comepelled to move more center if it's close election
                
                
                if cur_A < cur_B:
                    cur_A += self.LOSER_MVT
                    cur_B -= self.LOSER_MVT
                    
                else:
                    cur_A -= self.LOSER_MVT
                    cur_B += self.LOSER_MVT
                                    
                result = self.count_points([cur_A, cur_B])
                interp_df, opt_pos = interpolate([cur_A, cur_B], result, self.POS_MAX, ['A', 'B'])
                self.fill_results_dict(cur_A, cur_B, result[0], result[1])
                prev_movement = None
                continue            
            
            
            
            
            ########################
            
            loser = np.argmin(result)
            
            """
                defaults to moving towards your opponent.  
                ## however, if you lost last time,
                moved towards your opponent previously,
                and did worse, 
                then you should move away from opp.
            """    
            if loser == 0:
                loser_key = 'A'
                winner_key = 'B'
            else:
                loser_key = 'B'
                winner_key = 'A'            
            
            
            movement = self.LOSER_MVT
            
            if len(self.results_dict['A']) > 1 and prev_movement:
                
                prev_results = self.count_points([self.results_dict['A'][-2], self.results_dict['B'][-2]])
                prev_loser = np.argmin(result)
                
                
                
                ## if you lost last time and did worse this time
                if prev_loser == loser and prev_results[loser] > result[loser]:
                    movement = -prev_movement
                    prev_movement = movement
                
                
                
                else:
                    #move towards opponent
                    if self.results_dict[loser_key][-1]  > self.results_dict[winner_key][-1]:
                        movement = -self.LOSER_MVT
                        prev_movement = movement
                    
            ## if first or second move        
            else:
                #move towards opponent
                if self.results_dict[loser_key][-1]  > self.results_dict[winner_key][-1]:
                    movement = -self.LOSER_MVT                
                    prev_movement = movement
            

            

            if loser_key == 'A':
                cur_A += movement
                
            else:
                cur_B += movement

            result = self.count_points([cur_A, cur_B])
            interp_df, opt_pos = interpolate([cur_A, cur_B], result, self.POS_MAX, ['A', 'B'])
            self.fill_results_dict(cur_A, cur_B, result[0], result[1])
            
            
            
            ### if just oscillating
            if self.oscilating_candidate(self.results_dict['A']) and self.oscilating_candidate(self.results_dict['B']):
                break
            

        return result, cur_A, cur_B
    


    
    
def animate_election_mvt(voters, A, B, save_file):
    fig, ax = plt.subplots()
    ax.hist(voters, color='green')
    #x = np.arange(0, 100, 0.01)
    #line, = ax.plot(x, 1000*np.sin(x))

    ax.axvline(A[0], ls='--', color='blue', label='A Initial')
    ax.axvline(B[0], ls='--', color='red', label='B Initial')



    vl_A = ax.axvline(A[0], ls='-', color='blue', label='A')
    vl_B = ax.axvline(B[0], ls='-', color='red', label='B')
    ax.legend()
    def animate(i, A, B, vl_A, vl_B):

        i = i%len(A)    

        vl_A.set_xdata(A[i])
        vl_B.set_xdata(B[i])
        return vl_A, vl_B,

    ani = animation.FuncAnimation(
        fig, animate, frames=len(A), interval=1000, blit=False, repeat_delay=2000, fargs=(A, B, vl_A, vl_B))
    plt.show()
    
    if save_file:

    # Set up formatting for the movie files
    
        """
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Adam Shen'), bitrate=1800)
        ani.save(save_file+'.mp4', writer=writer)
        """
        ani.save(save_file+'.mp4')
    return ani


    
    
def animate_model(model, save_file=None):
    
    res_dict = model.results_dict
    A = res_dict['A']
    B = res_dict['B']
    voters = model.VOTERS
    
    animate_election_mvt(voters, A, B, save_file)
    
    
    
    
    
    
    
            

def smart_optimal_loc_predictor(A_pos, B_pos, A_votes, B_votes):
    
    pass
    
    
    
def main():
    
    
    
    ####################### Adjustable Parameters
    MIN_DISTANCE = 5
    ALLOW_CANDIDATE_FLIPPING = True
    MAX_VOTER_DIFF_RATIO = 0.05
    POS_MAX = 100
    MAX_ITERS = 20
    
    
    VOTING_THRESH = 0 / 100
    
    voters = make_voters()
    all_candidates = make_init_candidates()
    
    ## Change your inverse benefit function to something else
    benefit_func= (lambda c, v: 1/np.abs(v-c))
    
    
    A = np.random.uniform(0, 50)
    B = np.random.uniform(50, 100)
    
    A = 10
    B = 35
    #######################
    
    
    ## 0 VOTING THRESHOLD (everyone must votes)
    VOTING_THRESH = 0
    print('MARGINAL BENEFIT VOTING THRESHOLD =', VOTING_THRESH)
        
    print('MODEL 1')
    e_model = election_model_1(voters=voters, min_cand_dist=MIN_DISTANCE, voter_diff_stop_ratio=MAX_VOTER_DIFF_RATIO, pos_max=POS_MAX, voting_thresh=VOTING_THRESH, allow_cand_flipping=ALLOW_CANDIDATE_FLIPPING, max_iterations=MAX_ITERS)
    e_model.show_many_candidates(all_candidates)
    e_model.summarize_generate_candidate_movements(A, B)
    
    #animation
    animate_model(e_model, 'model_1_VOTING_THRESH = {}, A = {}, B = {}, Flipping = {}'.format(VOTING_THRESH, A, B, ALLOW_CANDIDATE_FLIPPING))
    
    print('MODEL 2')
    MAX_ITERS = 150
    e_model_2 = election_model_2(max_iterations=MAX_ITERS, voters=voters, min_cand_dist=MIN_DISTANCE, voter_diff_stop_ratio=MAX_VOTER_DIFF_RATIO, pos_max=POS_MAX, voting_thresh=VOTING_THRESH, loser_mvt=1)
    e_model_2.summarize_generate_candidate_movements(A, B)
    animate_model(e_model_2, 'model_2_VOTING_THRESH = {}, A = {}, B = {}, Flipping = {}'.format(VOTING_THRESH, A, B, ALLOW_CANDIDATE_FLIPPING))
    
    
    
        
    
    
    ## Non-Zero VOTING THRESHOLD (people will not vote if marginal benefit of one candidate vs. other is less than this)
    
    VOTING_THRESH = 15 / 100
    print('MARGINAL BENEFIT VOTING THRESHOLD =', VOTING_THRESH)
    
    print('MODEL 1')
    e_model = election_model_1(voters=voters, min_cand_dist=MIN_DISTANCE, voter_diff_stop_ratio=MAX_VOTER_DIFF_RATIO, pos_max=POS_MAX, voting_thresh=VOTING_THRESH, allow_cand_flipping=ALLOW_CANDIDATE_FLIPPING, max_iterations=MAX_ITERS)
    e_model.show_many_candidates(all_candidates)
    e_model.summarize_generate_candidate_movements(A, B)
    
    #animation
    animate_model(e_model, 'model_1_VOTING_THRESH = {}, A = {}, B = {}, Flipping = {}'.format(VOTING_THRESH, A, B, ALLOW_CANDIDATE_FLIPPING))
    
    print('MODEL 2')
    MAX_ITERS = 150
    e_model_2 = election_model_2(max_iterations=MAX_ITERS, voters=voters, min_cand_dist=MIN_DISTANCE, voter_diff_stop_ratio=MAX_VOTER_DIFF_RATIO, pos_max=POS_MAX, voting_thresh=VOTING_THRESH, loser_mvt=1)
    e_model_2.summarize_generate_candidate_movements(A, B)
    animate_model(e_model_2, 'model_2_VOTING_THRESH = {}, A = {}, B = {}, Flipping = {}'.format(VOTING_THRESH, A, B, ALLOW_CANDIDATE_FLIPPING))
    
    
    
    
    
    
    
if __name__ == '__main__':
    main()
            
            
            
            
            
            
            
            
            
            
            
        
    
    
    