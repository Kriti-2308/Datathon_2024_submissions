
const sidebar = document.getElementById("sidebar");
const sidebarToggle = document.getElementById("sidebarToggle");
const closeSidebar = document.getElementById("closeSidebar");
const mainContent = document.getElementById("mainContent");


sidebarToggle.addEventListener("click", () => {
    sidebar.style.left = "0";
    mainContent.style.marginLeft = "250px";
});


closeSidebar.addEventListener("click", () => {
    sidebar.style.left = "-250px";
    mainContent.style.marginLeft = "0";
});
