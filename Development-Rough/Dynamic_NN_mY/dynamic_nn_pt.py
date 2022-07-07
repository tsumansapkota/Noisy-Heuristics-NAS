import torch.nn as nn
import torch as torch
import numpy as np

# https://en.wikipedia.org/wiki/Box-drawing_character
# https://en.wikipedia.org/wiki/Arrow_(symbol)

def _get_hidden_neuron_number(i, o):
    
    #### get minimum of any
#     return min((i, o))+1

    #### get geometric mean
#     return np.sqrt(i*o)

    #### modified geometric mean
    return (max(i,o)*(min(i,o)**2))**(1/3)


class HierarchicalResidual(nn.Module):

    def __init__(self, tree, input_dim, output_dim):
        super().__init__()

        self.tree = tree
        self.input_dim = input_dim
        self.output_dim = output_dim
        ## this can be Shortcut Layer or None
        self.shortcut = Shortcut(tree, self.input_dim, self.output_dim) 
        self.tree.parent_dict[self.shortcut] = self
        
        self.residual = None ## this can be Residual Layer or None
        ##### only one of shortcut or residual can be None at a time
        self.forward = self.forward_shortcut
        
        self.std_ratio = 0. ## 0-> all variation due to shortcut, 1-> residual
        self.target_std_ratio = 0. ##
    
    def forward_both(self, x):
        s = self.shortcut(x)
        r = self.residual(x)

        if self.residual.hook is None: ### dont execute when computing significance
            s_std = torch.std(s, dim=0, keepdim=True)
            r_std = torch.std(r, dim=0, keepdim=True)
            stdr = r_std/(s_std+r_std)

            self.std_ratio = self.tree.beta_std_ratio*self.std_ratio + (1-self.tree.beta_std_ratio)*stdr.data
            if r_std.min() > 1e-9:
                ## recover for the fact that when decaying neurons, target ratio should also be reducing
                if self.tree.total_decay_steps:
                    i, o = self.shortcut.weight.shape[1],self.shortcut.weight.shape[0]
                    if self.shortcut.to_remove is not None:
                        i -= len(self.shortcut.to_remove)
                    if self.shortcut.to_freeze is not None:
                        o -= len(self.shortcut.to_freeze)
                    h = self.residual.hidden_dim
                    if self.residual.to_remove is not None:
                        h -= len(self.residual.to_remove)
                    
                    # tr = h/np.ceil((i+o)/2 +1)
                    tr = h/_get_hidden_neuron_number(i, o)

                    self.compute_target_std_ratio(tr)
                else:
                    self.compute_target_std_ratio()
                self.get_std_loss(stdr)
        return s+r
    
    def forward_shortcut(self, x):
        return self.shortcut(x)
    
    def forward_residual(self, x):
        self.compute_target_std_ratio()
        return self.residual(x)
    
    def compute_target_std_ratio(self, tr = None):
        if tr is None:
#             tr = self.residual.hidden_dim/np.ceil((self.input_dim+self.output_dim)/2 +1)
            tr = self.residual.hidden_dim/_get_hidden_neuron_number(self.input_dim, self.output_dim)
#             tr = self.residual.hidden_dim/np.ceil(self.output_dim/2 +1)

        tr = np.clip(tr, 0., 1.)
        self.target_std_ratio = self.tree.beta_std_ratio*self.target_std_ratio +\
                                (1-self.tree.beta_std_ratio)*tr
        pass        
    
    def get_std_loss(self, stdr):
        del_std = self.target_std_ratio-stdr
        del_std_loss = (del_std**2 + torch.abs(del_std)).mean()
#         del_std_loss = (del_std**2).mean()
        self.tree.std_loss += del_std_loss
        return
            
    def start_freezing_connection(self, to_freeze):
        if self.shortcut:
            self.shortcut.start_freezing_connection(to_freeze)
        if self.residual:
            self.residual.fc1.start_freezing_connection(to_freeze)
        pass
        
    def start_decaying_connection(self, to_remove):
        if self.shortcut:
            self.shortcut.start_decaying_connection(to_remove)
        if self.residual:
            self.residual.fc0.start_decaying_connection(to_remove)
        pass
    
    def remove_freezed_connection(self, remaining):
        if self.shortcut:
            self.shortcut.remove_freezed_connection(remaining)
        if self.residual:
            self.residual.fc1.remove_freezed_connection(remaining)
            if self.shortcut: self.std_ratio = self.std_ratio[:, remaining]
        self.output_dim = len(remaining)
        pass
    
    def remove_decayed_connection(self, remaining):
        if self.shortcut:
            self.shortcut.remove_decayed_connection(remaining)
        if self.residual:
            self.residual.fc0.remove_decayed_connection(remaining)
        self.input_dim = len(remaining)
        pass
    
    def add_input_connection(self, num):
        self.input_dim += num
        if self.shortcut:
            self.shortcut.add_input_connection(num)
        if self.residual:
            self.residual.fc0.add_input_connection(num)

    def add_output_connection(self, num):
        self.output_dim += num
        if self.shortcut:
            self.shortcut.add_output_connection(num)
        if self.residual:
            self.residual.fc1.add_output_connection(num)
            # if torch.is_tensor(self.std_ratio):
            if self.shortcut:
                self.std_ratio = torch.cat((self.std_ratio, torch.zeros(1, num)), dim=1)

    def add_hidden_neuron(self, num):
        if num<1: return
        
        if self.residual is None:
            # print(f"Adding {num} hidden units.. in new residual_layer")
            self.residual = Residual(self.tree, self.input_dim, num, self.output_dim)
            self.tree.parent_dict[self.residual] = self
            if self.shortcut is None:
                self.forward = self.forward_residual
                self.std_ratio = 1.
            else:
                self.forward = self.forward_both
                self.std_ratio = torch.zeros(1, self.output_dim)
                
        else:
            # print(f"Adding {num} hidden units..")
            self.residual.add_hidden_neuron(num)
        return
    
    def maintain_shortcut_connection(self):
        if self.residual is None: return
        
        if self.shortcut:
            if self.std_ratio.min()>0.98 and self.target_std_ratio>0.98:
                del self.tree.parent_dict[self.shortcut]
                del self.shortcut
                self.shortcut = None
                self.forward = self.forward_residual
                self.std_ratio = 1.
            
        elif self.target_std_ratio<0.95:
            self.shortcut = Shortcut(self.tree, self.input_dim, self.output_dim)
            self.shortcut.weight.data *= 0.
            self.forward = self.forward_both
            
        self.residual.fc0.maintain_shortcut_connection()
        self.residual.fc1.maintain_shortcut_connection()
        
    def morph_network(self):
        if self.residual is None: return
        
        if self.residual.hidden_dim < 1:
            del self.tree.parent_dict[self.residual]
            del self.residual
            ### its parent removes it from dynamic list if possible
            self.residual = None
            self.forward = self.forward_shortcut
            self.std_ratio = 0.
            return
        
#         max_dim = np.ceil((self.input_dim+self.output_dim)/2)
        # max_dim = min((self.input_dim, self.output_dim))+1
        max_dim = _get_hidden_neuron_number(self.input_dim, self.output_dim) + 1 
        # print("MaxDIM", max_dim, self.residual.hidden_dim)
        if self.residual.hidden_dim > max_dim:
            self.tree.DYNAMIC_LIST.add(self.residual.fc0)
            self.tree.DYNAMIC_LIST.add(self.residual.fc1)
            # print("Added", self.residual)
            
        # self.residual.fc0.morph_network()
        # self.residual.fc1.morph_network()
        self.residual.morph_network()
        
    def print_network_debug(self, depth):
        stdr = self.std_ratio
        if torch.is_tensor(self.std_ratio):
            stdr = self.std_ratio.min()
            
        print(f"{'|     '*depth}H:{depth}[{self.input_dim},{self.output_dim}]"+\
              f"σ[t:{self.target_std_ratio}, s:{stdr}")
        if self.shortcut:
            self.shortcut.print_network_debug(depth+1)
        if self.residual:
            self.residual.print_network_debug(depth+1)
        pass
    
    def print_network(self, pre_string=""):
        if self.residual is None:
            return
        
        if self.shortcut:
            print(f"{pre_string}├────┐")
            self.residual.print_network(f"{pre_string}│    ")
            print(f"{pre_string}├────┘")
        else:
            print(f"{pre_string}└────┐")
            self.residual.print_network(f"{pre_string}     ")
            print(f"{pre_string}┌────┘")
        return
        

class Shortcut(nn.Module):

    def __init__(self, tree, input_dim, output_dim):
        super().__init__()
        self.tree = tree
        _wd = nn.Linear(input_dim, output_dim, bias=False).weight.data
        self.weight = nn.Parameter(
            torch.empty_like(_wd).copy_(_wd)
        )
    
        ## for removing and freezing neurons
        self.to_remove = None
        self.to_freeze = None
        self.initial_remove = None
        self.initial_freeze = None
    
    def forward(self, x):
        ## input_dim        ## output_dim
        if x.shape[1] + self.weight.shape[1] > 0:
            return x.matmul(self.weight.t())
        else:
            # print(x.shape, self.weight.shape)
            # print(x.matmul(self.weight.t()))
            if x.shape[1] + self.weight.shape[1] == 0:
                return torch.zeros(x.shape[0], self.weight.shape[0])

    def decay_std_ratio(self, factor):
        self.weight.data = self.weight.data - self.tree.decay_rate_std*factor.t()*self.weight.data
        
    def decay_std_ratio_grad(self, factor):
        self.weight.grad = self.weight.grad + self.tree.decay_rate_std*factor.t()*self.weight.data
        
    def start_decaying_connection(self, to_remove):
        self.initial_remove = self.weight.data[:, to_remove]
        self.to_remove = to_remove
        self.tree.decay_connection_shortcut.add(self)
        pass
    
    def start_freezing_connection(self, to_freeze):
        self.initial_freeze = self.weight.data[to_freeze, :]
        self.to_freeze = to_freeze
        self.tree.freeze_connection_shortcut.add(self)
        pass
    
    def freeze_connection_step(self):#, to_freeze):
        self.weight.data[self.to_freeze, :] = self.initial_freeze
        pass
    
    def decay_connection_step(self):#, to_remove):
        self.weight.data[:, self.to_remove] = self.initial_remove*self.tree.decay_factor
        pass
            
     
    def remove_freezed_connection(self, remaining):
        # print(self.weight.data.shape, "removing freezed; ", self.to_freeze)
        _w = self.weight.data[remaining, :]
        del self.weight
        self.weight = nn.Parameter(_w)
        self.initial_freeze = None
        self.to_freeze = None
        pass
    
    def remove_decayed_connection(self, remaining):
        # print(self.weight.data.shape, "removing decayed; ", self.to_remove)
        _w = self.weight.data[:, remaining]
        del self.weight
        self.weight = nn.Parameter(_w)
        self.initial_remove = None
        self.to_remove = None
        pass
    
    def add_input_connection(self, num):
        # print(self.weight.data.shape)
        o, i = self.weight.data.shape
        _w = torch.cat((self.weight.data, torch.zeros(o, num)), dim=1)
        del self.weight
        self.weight = nn.Parameter(_w)
        # print(self.weight.data.shape)
        pass

    def add_output_connection(self, num):
        # print(self.weight.data.shape)
        o, i = self.weight.data.shape
        stdv = 1. / np.sqrt(i)
        _new = torch.empty(num, i, dtype=self.weight.data.dtype).uniform_(-stdv, stdv)
        
        _w = torch.cat((self.weight.data, _new), dim=0)
        del self.weight
        self.weight = nn.Parameter(_w)
        # print(self.weight.data.shape)        
        pass
    
    def print_network_debug(self, depth):
        print(f"{'|     '*depth}S:{depth}[{self.weight.data.shape[1]},{self.weight.data.shape[0]}]")

    
class Residual(nn.Module):

    def __init__(self, tree, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.tree = tree
        self.hidden_dim = hidden_dim
        
        self.del_neurons = 0.
        self.neurons_added = 0

        ## Shortcut or Hierarchical Residual Layer
        self.fc0 = HierarchicalResidual(self.tree, input_dim, hidden_dim) 
        self.non_linearity = NonLinearity(self.tree, hidden_dim)
        self.fc1 = HierarchicalResidual(self.tree, hidden_dim, output_dim)
        self.fc1.shortcut.weight.data *= 0.
        
        self.tree.parent_dict[self.fc0] = self
        self.tree.parent_dict[self.fc1] = self
        self.tree.parent_dict[self.non_linearity] = self
        
        self.hook = None
        self.significance = None
        self.count = None
        self.apnz = None
        self.to_remove = None
    
    def forward(self, x):
        h = self.fc0(x)
        self.activations = self.non_linearity(h)
        h = self.fc1(self.activations)
        return h
    
    def start_computing_significance(self):
        self.significance = 0.
        self.count = 0
        self.apnz = 0
        self.hook = self.non_linearity.register_backward_hook(self.compute_neuron_significance)
        pass
            
    def finish_computing_significance(self):
        self.hook.remove()
        self.significance = self.significance/self.count
        self.apnz = self.apnz/self.count
        self.significance = self.significance*(1-self.apnz) * 4 ## tried on desmos.
        self.count = None
        self.hook = None
        pass
    
    def compute_neuron_significance(self, _class, grad_input, grad_output):
        with torch.no_grad():
            # self.significance += torch.sum(torch.abs(grad_output[0].data*self.activations.data), dim=0)
            self.significance += torch.sum((grad_output[0].data*self.activations.data)**2, dim=0)
            self.count += grad_output[0].shape[0]
    #         self.apnz += torch.count_nonzero(self.activations.data, dim=0)
            self.apnz += torch.sum(self.activations.data > 0., dim=0, dtype=torch.float)
        pass
    
    def identify_removable_neurons(self, below):
        if self.to_remove is not None:
            print("First remove all previous less significant neurons")
            return
        
        self.to_remove = torch.nonzero(self.significance<=below).reshape(-1)
        if len(self.to_remove)>0:
            self.fc0.start_freezing_connection(self.to_remove)
            self.fc1.start_decaying_connection(self.to_remove)
            self.tree.remove_neuron_residual.add(self)
            return len(self.to_remove)
        
        self.to_remove = None
        return 0

    def remove_decayed_neurons(self):
        remaining = []
        for i in range(self.hidden_dim):
            if i not in self.to_remove:
                remaining.append(i)
        
        self.non_linearity.remove_neuron(remaining)
        self.fc0.remove_freezed_connection(remaining)
        self.fc1.remove_decayed_connection(remaining)
        
        self.neurons_added -= len(self.to_remove)
        self.hidden_dim = len(remaining)
        self.to_remove = None
        pass
    
    def compute_del_neurons(self):
        self.del_neurons = (1-self.tree.beta_del_neuron)*self.neurons_added \
                            + self.tree.beta_del_neuron*self.del_neurons
        self.neurons_added = 0
        return
    
    def add_hidden_neuron(self, num):
        self.fc0.add_output_connection(num)
        self.non_linearity.add_neuron(num)
        self.fc1.add_input_connection(num)
        
        self.hidden_dim += num
        self.neurons_added += num
        pass

    def morph_network(self):
        self.fc0.morph_network()
        self.fc1.morph_network()
#         max_dim = np.ceil((self.tree.parent_dict[self].input_dim+\
#             self.tree.parent_dict[self].output_dim)/2)
        max_dim = _get_hidden_neuron_number(self.tree.parent_dict[self].input_dim,
            self.tree.parent_dict[self].output_dim)+1
        if self.hidden_dim <= max_dim:
            if self.fc0.residual is None:
                if self.fc0 in self.tree.DYNAMIC_LIST:
                    self.tree.DYNAMIC_LIST.remove(self.fc0)
            if self.fc1.residual is None:
                if self.fc1 in self.tree.DYNAMIC_LIST:
                    self.tree.DYNAMIC_LIST.remove(self.fc1)
        return 

    def print_network_debug(self, depth):
        print(f"{'|     '*depth}R:{depth}[{self.hidden_dim}|{self.non_linearity.bias.data.shape[0]}]")
        self.fc0.print_network_debug(depth+1)
        self.fc1.print_network_debug(depth+1)
        
    def print_network(self, pre_string):
        self.fc0.print_network(pre_string)
        print(f"{pre_string}{self.hidden_dim}")
        self.fc1.print_network(pre_string)
        return


class NonLinearity(nn.Module):

    def __init__(self, tree, io_dim, actf_obj=nn.ReLU()):
        super().__init__()
        self.tree = tree
        self.bias = nn.Parameter(torch.zeros(io_dim))
        self.actf = actf_obj

    def forward(self, x):
        return self.actf(x+self.bias)

    def add_neuron(self, num):
        _b = torch.cat((self.bias.data, torch.zeros(num)))
        del self.bias
        self.bias = nn.Parameter(_b)
        
    def remove_neuron(self, remaining):
        _b = self.bias.data[remaining]
        del self.bias
        self.bias = nn.Parameter(_b)

    
class Tree_State():
    
    def __init__(self):
        self.DYNAMIC_LIST = set() ## residual parent is added, to make code effecient.
        ## the parents which is not intended to have residual connection should not be added.
        self.beta_std_ratio = None
        self.beta_del_neuron = None
    
        self.parent_dict = {}
    
        self.total_decay_steps = None
        self.current_decay_step = None
        self.decay_factor = None
        self.remove_neuron_residual:set = None
        self.freeze_connection_shortcut:set = None
        self.decay_connection_shortcut:set = None

        self.decay_rate_std = 0.001

        self.add_to_remove_ratio = 2.
        pass
    
    def get_decay_factor(self):
        ratio = self.current_decay_step/self.total_decay_steps
#         self.decay_factor = np.exp(-2*ratio)*(1-ratio)
        self.decay_factor = (1-ratio)**2
        pass
    
    def clear_decay_variables(self):
        self.total_decay_steps = None
        self.current_decay_step = None
        self.decay_factor = None
        self.remove_neuron_residual = None
        self.freeze_connection_shortcut = None
        self.decay_connection_shortcut = None
        
    

class Dynamic_Network(nn.Module):

    def __init__(self, input_dim, output_dim, final_activation=None,
                 num_stat=5, num_std=100, decay_rate_std=0.001):
        super().__init__()
        self.tree = Tree_State()
        self.tree.beta_del_neuron = (num_stat-1)/num_stat
        self.tree.beta_std_ratio = (num_std-1)/num_std
        self.tree.decay_rate_std = decay_rate_std
        
        self.root_net = HierarchicalResidual(self.tree, input_dim, output_dim)
        self.tree.DYNAMIC_LIST.add(self.root_net)
        self.tree.parent_dict[self.root_net] = None
        
        if final_activation is None:
            final_activation = lambda x: x
        self.non_linearity = NonLinearity("Root", output_dim, final_activation)
        
        self.neurons_added = 0

        self._remove_below = None ## temporary variable
    
    def forward(self, x):
        return self.non_linearity(self.root_net(x))


    def add_neurons(self, num):
        num_stat = num//2
        num_random = num - num_stat
        
        DL = list(self.tree.DYNAMIC_LIST)
        if num_random>0:
            rands = torch.randint(high=len(DL), size=(num_random,))
            index, count = torch.unique(rands, sorted=False, return_counts=True)
            for i, idx in enumerate(index):
                DL[idx].add_hidden_neuron(int(count[i]))

        if num_stat>0:
            del_neurons = []
            for hr in DL:
                if hr.residual:
                    del_neurons.append(hr.residual.del_neurons)#+1e-7)
                else:
                    del_neurons.append(0.)#1e-7) ## residual layer yet not created 
            
            prob_stat = torch.tensor(del_neurons)
            prob_stat = torch.log(torch.exp(prob_stat)+1.)
            m = torch.distributions.multinomial.Multinomial(total_count=num_stat,
                                                            probs= prob_stat)
            count = m.sample()#.type(torch.long)
            for i, hr in enumerate(DL):
                if count[i] < 1: continue
                hr.add_hidden_neuron(int(count[i]))
        
        self.neurons_added += num 
        pass

    def identify_removable_neurons(self, num=None, threshold_min=0., threshold_max=1.):
        
        all_sig = []
        self.all_sig_ = []
        for hr in self.tree.DYNAMIC_LIST:
            if hr.residual:
                all_sig.append(hr.residual.significance)
        all_sigs = torch.cat(all_sig)

        max_sig = all_sigs.max()
        all_sig = all_sigs/(max_sig+1e-9)

        all_sig = all_sig[all_sig<threshold_max]
        if len(all_sig)<1:
            return 0, None, all_sigs
        all_sig = torch.sort(all_sig)[0] ### sorted significance scores
        
        self.all_sig_ = all_sig
        
        if not num:num = int(np.ceil(self.neurons_added/self.tree.add_to_remove_ratio))
        ## reset the neurons_added number if decay is started

        remove_below = threshold_min
        if num>len(all_sig):
            remove_below = all_sig[-1]
        elif num>0:
            remove_below = all_sig[num-1]

        remove_below *= max_sig

        self._remove_below = remove_below
        return remove_below, all_sigs

    def decay_neuron_start(self, decay_steps=1000):
        if self._remove_below is None: return 0
        
        self.neurons_added = 0 ## resetting this variable
        
        self.tree.total_decay_steps = decay_steps
        self.tree.current_decay_step = 0
        self.tree.remove_neuron_residual = set()
        self.tree.freeze_connection_shortcut = set()
        self.tree.decay_connection_shortcut = set()
        
        count_remove = 0
        for hr in self.tree.DYNAMIC_LIST:
            if hr.residual:
                count_remove += hr.residual.identify_removable_neurons(below=self._remove_below)
        if count_remove<1:
            self.tree.clear_decay_variables()
        return count_remove
    
    def decay_neuron_step(self):
        if self.tree.total_decay_steps is None:
            return
        
        self.tree.current_decay_step += 1
        
        if self.tree.current_decay_step < self.tree.total_decay_steps:
            self.tree.get_decay_factor()
            for sh in self.tree.decay_connection_shortcut:
                sh.decay_connection_step()
            for sh in self.tree.freeze_connection_shortcut:
                sh.freeze_connection_step()
        else:
            for rs in self.tree.remove_neuron_residual:
                rs.remove_decayed_neurons()
            
            self.tree.clear_decay_variables()
            
    def compute_del_neurons(self):
        for hr in self.tree.DYNAMIC_LIST:
            if hr.residual:
                hr.residual.compute_del_neurons()
    
    def maintain_network(self):
        self.root_net.maintain_shortcut_connection()
        self.root_net.morph_network()
        
    def start_computing_significance(self):
        for hr in self.tree.DYNAMIC_LIST:
            if hr.residual:
                hr.residual.start_computing_significance()

    def finish_computing_significance(self):
        for hr in self.tree.DYNAMIC_LIST:
            if hr.residual:
                hr.residual.finish_computing_significance()
            
    def print_network_debug(self):
        self.root_net.print_network_debug(0)
        
    def print_network(self):
        print(self.root_net.input_dim)
        self.root_net.print_network()
        print("│")
        print(self.root_net.output_dim)
        return