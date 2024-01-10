import torch
from util.util import copy_index

def recursive_ray_mesh(
        self, rays, model=False, n=1.5, reflection=False, refraction=False, max_rf=2
):
        face_normals = self.face_normals
        mesh = self.mesh
        num_rf = 0
        input_rays = rays
        while num_rf < max_rf:
            num_rf += 1
            if num_rf > max_rf: break
            elif num_rf == max_rf:
                far_flag = True
            else:
                far_flag = False

            (valid_mask,intersection_depth,normals,intersection_face,
            ) = self.intersection_with_mesh(input_rays, mesh, far=far_flag)

            if len(intersection_face) == 0:
                normals = face_normals[[]]
            else:
                normals = face_normals[intersection_face]                  
            normals = normals.to(device=rays.device)
            valid_mask = valid_mask.to(device=rays.device)
            hit_rays = rays[valid_mask]
            z_depth = intersection_depth[valid_mask].to(device=rays.device)
            if num_rf == 1: 
                init_depth = z_depth
                refr_rays[:, 7] = rays[valid_mask][:, 7]
                if reflection: #reflection, only consider the first intersection
                    refl_rays = torch.zeros_like(hit_rays)
                    refl_rays[:, :3] = hit_pts
                    if self.stage == 3:
                        model_input = torch.cat([hit_pts, normals], dim=-1)
                        if len(intersection_face) == 0:
                            normals_ = torch.zeros_like(hit_pts)
                        else:
                            normals_ = model(model_input)
                            normals_ = normals_ / (
                                torch.norm(normals_, dim=-1).unsqueeze(-1) + 1e-6
                            )
                        dir_delta = torch.acos(
                            (normals * normals_).sum(-1) / (torch.norm(normals, dim=-1) + 1e-6)
                        )
                    refl_dir = in_dir + 2 * cos_i.unsqueeze(-1) * (
                        normals if self.stage != 3 else normals_
                    )
                    refl_dir = refl_dir / torch.norm(refl_dir, dim=-1).unsqueeze(-1)
                    refl_rays[:, 3:6] = refl_dir
                    refl_rays[:, 6:] = rays[valid_mask][:, 6:]

                    if not refraction:
                        return (
                            rays,
                            valid_mask,
                            None,
                            refl_rays,
                            None,
                            (None if self.stage != 3 else dir_delta),
                        )                

            hit_pts = hit_rays[..., :3] + z_depth.unsqueeze(1) * hit_rays[..., 3:6]
            in_dir = hit_rays[..., 3:6]
            normals = self.adjust_normal(normals, in_dir)
            cos_i = (-in_dir * normals).sum(dim=1)
            n_ = 1 / n
            refr_rays = torch.zeros_like(hit_rays)
            refr_rays[:, :3] = hit_pts
            cos_i = (-in_dir * normals).sum(dim=1)


            if num_rf % 2 == 1: # from air to glass
                tmp = 1 - (n_**2) * (1 - cos_i**2)
                # tmp[tmp < 0] = 0.0
                cos_o = torch.sqrt(tmp)
                if self.stage == 3:
                    raise NotImplementedError
                refr_dir = n_ * in_dir + (n_ * cos_i - cos_o).unsqueeze(-1) * normals
                refr_dir = refr_dir / torch.norm(refr_dir, dim=-1).unsqueeze(-1)
                refr_rays[:, 3:6] = refr_dir
                input_rays = refr_rays
                fresnel_1 = self.Fresnel_term(n, in_dir, refr_dir, normals)
            else: # from glass to air
                tmp = 1 - (n**2) * (1 - cos_i**2)
                tmp[tmp < 0] = 0.0
                cos_o = torch.sqrt(tmp)                               
                if self.stage == 3:
                    model_input = torch.cat([in_dir, hit_pts, normals_], dim=-1)
                    if len(intersection_face) == 0:
                        dir_delta = torch.zeros_like(hit_pts)
                    else:
                        dir_delta = model(model_input)
                refr_dir_ = (n * in_dir+ (n * cos_i - cos_o).unsqueeze(-1) * normals_+ (dir_delta if self.stage == 3 else 0))
                refr_dir_ = refr_dir_ / torch.norm(refr_dir_, dim=-1).unsqueeze(-1)
                refr_rays[:, 3:6] = refr_dir_
                input_rays = refr_rays
            # dealing with rays missing the second intersection (unwatertight surface)
            invalid_mask_ = (valid_mask == False).to(device=rays.device)
            refr_rays = copy_index(refr_rays, invalid_mask_, refr_rays[invalid_mask_])                

            # update depth: intersection with planes
            # Is the init_depth the 1st or previous? what is the depth meaning?
            refr_rays[:, 7] = refr_rays[:, 7] - z_depth

        new_depth = init_depth

        out_rays = copy_index(rays, valid_mask, refr_rays)
        fresnel = 1 - fresnel_1
        return (
            out_rays,
            valid_mask,
            new_depth,
            None,
            fresnel,
            (None if self.stage != 3 else dir_delta),
        )

























def trace_ray_mesh(
        self, rays, model=False, n=1.5, reflection=False, refraction=False
    ):
        """
        Traces rays and performs reflection and refraction when hitting the surface.
        :param rays: [N,8], depths: [N,1], normals: [N,3]
        """
        face_normals = self.face_normals
        mesh = self.mesh
        # first intersection
        (
            valid_mask,
            intersection_depth,
            normals,
            intersection_face,
        ) = self.intersection_with_mesh(rays, mesh, far=False)
        if len(intersection_face) == 0:
            normals = face_normals[[]]
        else:
            normals = face_normals[intersection_face]
        normals = normals.to(device=rays.device)
        valid_mask = valid_mask.to(device=rays.device)
        hit_rays = rays[valid_mask]
        init_depth = intersection_depth[valid_mask].to(device=rays.device)
        hit_pts = hit_rays[..., :3] + init_depth.unsqueeze(1) * hit_rays[..., 3:6]
        in_dir = hit_rays[..., 3:6]
        normals = self.adjust_normal(normals, in_dir)
        cos_i = (-in_dir * normals).sum(dim=1)

        # reflection
        if reflection:
            refl_rays = torch.zeros_like(hit_rays)
            refl_rays[:, :3] = hit_pts
            if self.stage == 3:
                model_input = torch.cat([hit_pts, normals], dim=-1)
                if len(intersection_face) == 0:
                    normals_ = torch.zeros_like(hit_pts)
                else:
                    normals_ = model(model_input)
                    normals_ = normals_ / (
                        torch.norm(normals_, dim=-1).unsqueeze(-1) + 1e-6
                    )
                dir_delta = torch.acos(
                    (normals * normals_).sum(-1) / (torch.norm(normals, dim=-1) + 1e-6)
                )
            refl_dir = in_dir + 2 * cos_i.unsqueeze(-1) * (
                normals if self.stage != 3 else normals_
            )
            refl_dir = refl_dir / torch.norm(refl_dir, dim=-1).unsqueeze(-1)
            refl_rays[:, 3:6] = refl_dir
            refl_rays[:, 6:] = rays[valid_mask][:, 6:]

            if not refraction:
                return (
                    rays,
                    valid_mask,
                    None,
                    refl_rays,
                    None,
                    (None if self.stage != 3 else dir_delta),
                )

        # first refraction
        if self.stage == 3:
            raise NotImplementedError
        refr_rays = torch.zeros_like(hit_rays)
        refr_rays[:, :3] = hit_pts
        n_ = 1 / n
        cos_o = torch.sqrt(1 - (n_**2) * (1 - cos_i**2))
        refr_dir = n_ * in_dir + (n_ * cos_i - cos_o).unsqueeze(-1) * normals
        refr_dir = refr_dir / torch.norm(refr_dir, dim=-1).unsqueeze(-1)
        refr_rays[:, 3:6] = refr_dir
        fresnel_1 = self.Fresnel_term(n, in_dir, refr_dir, normals)

        # second intersection
        (
            valid_mask_,
            intersection_depth_,
            normals_,
            intersection_face_,
        ) = self.intersection_with_mesh(refr_rays, mesh, far=True)
        normals_ = normals_.to(device=rays.device)
        if len(intersection_face_) != 0:
            normals_[valid_mask_] = face_normals[intersection_face_]
        else:
            normals_[valid_mask_] = face_normals[[]]
        zdepth = intersection_depth_.to(device=rays.device)
        normals_ = normals_.to(device=rays.device)
        hit_pts_ = refr_rays[..., :3] + zdepth.unsqueeze(1) * refr_rays[..., 3:6]
        in_dir_ = refr_rays[..., 3:6]
        normals_ = self.adjust_normal(normals_, in_dir_)

        # second refraction
        refr_rays_ = torch.zeros_like(hit_rays)
        refr_rays_[:, :3] = hit_pts_
        cos_i = (-in_dir_ * normals_).sum(dim=1)
        tmp = 1 - (n**2) * (1 - cos_i**2)
        tmp[tmp < 0] = 0.0
        cos_o = torch.sqrt(tmp)
        if self.stage == 3:
            model_input = torch.cat([in_dir_, hit_pts_, normals_], dim=-1)
            if len(intersection_face_) == 0:
                dir_delta = torch.zeros_like(hit_pts_)
            else:
                dir_delta = model(model_input)
        refr_dir_ = (
            n * in_dir_
            + (n * cos_i - cos_o).unsqueeze(-1) * normals_
            + (dir_delta if self.stage == 3 else 0)
        )
        refr_dir_ = refr_dir_ / torch.norm(refr_dir_, dim=-1).unsqueeze(-1)
        refr_rays_[:, 3:6] = refr_dir_

        # dealing with rays missing the second intersection (unwatertight surface)
        invalid_mask_ = (valid_mask_ == False).to(device=rays.device)
        refr_rays_ = copy_index(refr_rays_, invalid_mask_, refr_rays[invalid_mask_])

        # update depth: intersection with planes
        refr_rays_[:, 7] = rays[valid_mask][:, 7] - init_depth - zdepth
        new_depth = init_depth

        out_rays = copy_index(rays, valid_mask, refr_rays_)
        fresnel = 1 - fresnel_1
        return (
            out_rays,
            valid_mask,
            new_depth,
            None,
            fresnel,
            (None if self.stage != 3 else dir_delta),
        )

