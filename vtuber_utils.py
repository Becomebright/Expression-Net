import os
import numpy as np
import scipy.io
import utils


# Convert 3DMM expression coefficients to blendshape coefficients
def convert_expr_coeffs(coeffs_dict, expr):
    # B = E * C
    CP= np.load('convert_matrix.npy')  # (29, 62)
    CN = np.load('convert_matrix_negative.npy')  # (29, 62)
    E = np.reshape(expr, (1, 29))  # (1, 29)
    EP = np.maximum(E, 0)  # coefficients that > 0
    EP /= 3.0
    # print(EP)
    EN = np.minimum(E, 0)  # coefficients that < 0
    EN /= -2.0
    # print(EN)
    B = EP.dot(CP) + EN.dot(CN)
    B = B.flatten()  # (62, )
    # print(B.shape)
    for i, b in enumerate(B):
        coeffs_dict[str(i) + '_EXP'].append(b)

    np.save('./tmp/EP', EP)
    np.save('./tmp/EN', EN)

    return coeffs_dict


def get_expr_basis(weight):
    basis_folder = './exp_basis_' + str(weight) + '/'
    if not os.path.exists(basis_folder):
        os.makedirs(basis_folder)

    BFM_path = './Shape_Model/BaselFaceModel_mod.mat'  # 140MB
    model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
    model = model["BFM"]
    faces = model.faces - 1

    # Write the mean face
    S = model.expMU
    num_vert = int(S.shape[0] / 3)
    S = np.reshape(S, (num_vert, 3))
    mesh_name = basis_folder + 'mean_face.ply'
    print('write to: ' + mesh_name)
    utils.write_ply_textureless(mesh_name, S, faces)

    for i in range(29):
        expr = model.expEV * 0
        expr[i] = model.expEV[i] * weight
        E = np.matmul(model.expPC, expr)
        S = model.shapeMU + model.expMU + E
        num_vert = int(S.shape[0] / 3)
        S = np.reshape(S, (num_vert, 3))
        mesh_name = basis_folder + 'basis_' + str(i) + '.ply'
        print('write to: ' + mesh_name)
        utils.write_ply_textureless(mesh_name, S, faces)


# def write_meshes(frames, SEPs, faces):
#     start = time.time()
#     mesh_folder = './output_ply'  # The location where .ply files are saved
#     if not os.path.exists(mesh_folder):
#         os.makedirs(mesh_folder)
#
#     for i, SEP in enumerate(SEPs):
#         frame_name = mesh_folder + '/' + str(i) + '.jpg'
#         cv2.imwrite(frame_name, frames[i])
#         mesh_name = mesh_folder + '/' + str(i)
#         utils.write_ply_textureless(mesh_name + '_Shape_Expr_Pose.ply', SEP, faces)
#     print("Writing meshes complete {}".format(time.time() - start))
