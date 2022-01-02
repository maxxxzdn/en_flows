import torch
from qm9 import analyze

# bond length lists as torch tensors
bonds1_ = torch.tensor([list(x.values()) for x in analyze.bonds1.values()])
bonds2_ = torch.tensor([list(x.values()) for x in analyze.bonds2.values()])
bonds3_ = torch.tensor([list(x.values()) for x in analyze.bonds3.values()])

# parameters of the bond stretching term
kbIJ = torch.load('qm9/mmff_params/kb')
b0IJ = torch.load('qm9/mmff_params/b0')
# parameters of the van der Waals term
aI = torch.load('qm9/mmff_params/aI')
AI = torch.load('qm9/mmff_params/AI')
GI = torch.load('qm9/mmff_params/GI')
NI = torch.load('qm9/mmff_params/NI')


def pars_to_device(device):
    global kbIJ, b0IJ, aI, AI, GI, NI, bonds1_, bonds2_, bonds3_
    kbIJ = kbIJ.to(device)
    b0IJ = b0IJ.to(device)
    aI = aI.to(device)
    AI = AI.to(device)
    GI = GI.to(device)
    NI = NI.to(device)
    bonds1_ = bonds1_.to(device)
    bonds2_ = bonds2_.to(device)
    bonds3_ = bonds3_.to(device)


def get_bond_order_(positions, atom_type):
    """
    Parallel computation a bond order for the whole batch
    """
    assert atom_type.ndim == 3
    distance = torch.cdist(positions, positions, 2)
    distance = 100 * distance

    crit1 = (distance < bonds1_[atom_type, atom_type.permute(
        0, 2, 1)] + analyze.margin1).long()
    crit2 = (distance < bonds2_[atom_type, atom_type.permute(
        0, 2, 1)] + analyze.margin2).long()
    crit3 = (distance < bonds3_[atom_type, atom_type.permute(
        0, 2, 1)] + analyze.margin3).long()
    return crit1 + crit2 + crit3


def remove_diag(tensor):
    return tensor.tril(-1) + tensor.tril(-1).permute(0, 2, 1)


def get_atom_mask(positions):
    bs, n_nodes, _ = positions.shape
    p = positions.sum(-1).unsqueeze(-1).expand(bs, n_nodes, n_nodes) != 0
    return torch.minimum(p, p.permute(0, 2, 1))


def energy_bond(positions, atom_type, bond_type):
    """
    Bond stretching term calculated as follows:
        E_ij = k*(r_i - r_j)**2
    see mmff_params/mmff_params.html for details.
    """
    bond_ind = torch.where(bond_type > 0, bond_type - 1, 0)

    b0 = b0IJ[bond_ind, atom_type, atom_type.permute(0, 2, 1)]
    kb_ij = kbIJ[bond_ind, atom_type, atom_type.permute(0, 2, 1)]

    b = torch.cdist(positions, positions, 2)
    dr_ij = b - b0

    eb_ij = 0.5 * 143.9325 * kb_ij * dr_ij**2
    # only values for actual bond should remain
    e_bond = torch.where(bond_type > 0, eb_ij,
                         torch.zeros_like(eb_ij)).sum([-1, -2])
    return torch.clip(e_bond, None, 10)  # clip values to avoid outliers

def get_eps_ij(atom_type, r_IJ):
    """
    Returns depth of the energy well corresponding to IJ
    """
    return 181.16 * GI[atom_type] * GI[atom_type.permute(0, 2, 1)] * aI[atom_type] * aI[atom_type.permute(0, 2, 1)] / (
        (aI[atom_type] / NI[atom_type])**0.5 + (aI[atom_type.permute(0, 2, 1)] / NI[atom_type.permute(0, 2, 1)])**0.5) * r_IJ**(-6)


def energy_vdW(positions, atom_type, atom_mask):
    """
    van der Waals term calculated as follows:
        Evdw_ij  =  eIJ*{1.12R*IJ/(Rij+0.12R*IJ)}**7 *
                 {1.07 R*IJ**7/(Rij**7 + 0.07R*IJ**7) - 2}
    see mmff_params/mmff_params.html for details.
    """
    r_ij = torch.cdist(positions, positions, 2)
    r_II = AI[atom_type] * aI[atom_type]**(0.25)
    r_JJ = AI[atom_type.permute(0, 2, 1)] * \
        aI[atom_type.permute(0, 2, 1)]**(0.25)
    r_IJ = 0.5 * (r_II + r_JJ)
    eps_ij = get_eps_ij(atom_type, r_IJ)

    evdW_ij = eps_ij * (1.07 * r_IJ / (r_ij + 0.07 * r_IJ))**7 * \
        (1.12 * r_IJ**7 / (r_ij**7 + 0.12 * r_IJ**7) - 2)
    
    evdW_ij = remove_diag(evdW_ij)
    # only values for actual bond should remain
    evdW = torch.where(atom_mask > 0, evdW_ij,
                       torch.zeros_like(evdW_ij)).sum([-1, -2])
    return torch.clip(evdW, None, 10)  # clip values to avoid outliers


def energy_loss(x, one_hot):
    """
    potential energy calculated as sum of bond stretching and van der Waals terms
    """
    atom_type = torch.argmax(one_hot.float(), -1).unsqueeze(1)
    bond_type = remove_diag(get_bond_order_(x, atom_type))
    atom_mask = get_atom_mask(x)

    e_b = energy_bond(x, atom_type, bond_type)
    e_vdW = energy_vdW(x, atom_type, atom_mask)

    return e_b + e_vdW


def compute_energy_loss(dequantizer, flow, prior,
                        nodes_dist, x, node_mask, edge_mask, context):

    pars_to_device(x.device)
    bs, n_nodes, n_dims = x.size()
    edge_mask = edge_mask.view(bs, n_nodes * n_nodes)

    # sample from prior distribution and generate molecules
    z_x, z_h = prior.sample(bs, n_nodes, node_mask)
    z = torch.cat([z_x, z_h], dim=2)

    z = flow.reverse(z, node_mask, edge_mask, context)

    if torch.any(torch.isnan(z)).item() or torch.any(torch.isinf(z)).item():
        print('NaN occured, setting z to zero.')
        z = torch.zeros_like(z)

    x = z[:, :, 0:3]
    one_hot = z[:, :, 3:8]
    charges = z[:, :, 8:]

    tensor = dequantizer.reverse({'categorical': one_hot, 'integer': charges})
    one_hot, charges = tensor['categorical'], tensor['integer']

    return energy_loss(x.contiguous(), one_hot.contiguous()).mean()
