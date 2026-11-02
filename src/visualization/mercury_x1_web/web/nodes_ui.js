// ROS 2 node bubbles shared by the control page (:8080, Status tab) and the head display (:8081):
// expected nodes per group (green = running, red dashed = missing while its group should run), plus "Other".
(function () {
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // [group, node names, should run (procs = {slam, nav})]
  const GROUPS = [
    ["Drivers & sensors", ["mercury_base", "mercury_upper_body", "robot_state_publisher", "lslidar_driver_node", "camera",
      "camera_container", "bellybutton_camera", "unitree_remote"], () => true],
    ["Perception (GPU)", ["nvblox_node", "nvblox_mesh_tiler", "cuvslam_rgbd"], (p) => p.slam.running],
    ["SLAM", ["rtabmap", "icp_odometry", "rgbd_sync"], (p) => p.slam.running],
    ["Navigation", ["nav2_container", "controller_server", "planner_server", "bt_navigator", "behavior_server", "smoother_server",
      "velocity_smoother", "waypoint_follower", "lifecycle_manager_navigation", "local_costmap", "global_costmap",
      "arm_footprint"], (p) => p.nav.running],
    ["Arm planning", ["move_group"], (p) => !!(p.moveit && p.moveit.running)],
    ["Web & viewer", ["mercury_web", "rerun_bridge", "preview_orbbec", "preview_bellybutton"], () => true],
  ];

  // -> {html, running, missing}
  function render(nodes, procs) {
    const up = new Set(nodes || []), known = new Set(GROUPS.flatMap((g) => g[1]));
    let missing = 0;
    const html = GROUPS.map(([name, list, needed]) => {
      const need = needed(procs);
      return `<div class="group"><div class="gname">${esc(name)}</div><div class="bubbles">` +
        list.map((n) => {
          if (!up.has(n) && need) missing++;
          return `<span class="bubble ${up.has(n) ? "up" : "down" + (need ? " need" : "")}">${esc(n)}</span>`;
        }).join("") + "</div></div>";
    });
    const other = (nodes || []).filter((n) => !known.has(n));
    if (other.length) html.push(`<div class="group"><div class="gname">Other</div><div class="bubbles">` +
      other.map((n) => `<span class="bubble up">${esc(n)}</span>`).join("") + "</div></div>");
    return { html: html.join(""), running: (nodes || []).length, missing };
  }

  window.NodesUI = { GROUPS, render };
})();
