<script setup>
import MinCultLogo from "@/components/Footer/MinCultLogo.vue";
</script>

<template>
    <div class="footer">
        <div class="footer__content">
            <MinCultLogo/>
        </div>
    </div>
</template>

<style scoped>
.footer {
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-top: auto;
    height: 8.3125em;
    padding-top: 1.5625em;
    padding-bottom: 1.5625em;
    background-color: var(--fifth-color);
}
.footer__content {
    margin-left: 1.25em;
}
</style>