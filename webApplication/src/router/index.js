import { createMemoryHistory, createRouter } from 'vue-router'

import Search from "@/pages/Search.vue";
import Results from "@/pages/Results.vue";
import ExhibitDetails from "@/pages/ExhibitDetails.vue";

const routes = [
  { path: '/', component: Search, name: "Search" },
  { path: '/results', component: Results, name: "Results" },
  { path: '/exhibits/:id', component: ExhibitDetails, name: "Details" },
]

const router = createRouter({
  history: createMemoryHistory(),
  routes,
})

export default router;