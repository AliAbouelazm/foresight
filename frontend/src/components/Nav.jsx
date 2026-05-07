import { NavLink } from "react-router-dom";
import s from "../styles/components.module.css";

export default function Nav() {
  return (
    <nav className={s.nav}>
      <span className={s.navBrand}>Foresight</span>
      <div className={s.navLinks}>
        <NavLink
          to="/demo"
          className={({ isActive }) => `${s.navLink} ${isActive ? s.navLinkActive : ""}`}
        >
          Demo
        </NavLink>
        <NavLink
          to="/metrics"
          className={({ isActive }) => `${s.navLink} ${isActive ? s.navLinkActive : ""}`}
        >
          Metrics
        </NavLink>
      </div>
    </nav>
  );
}
